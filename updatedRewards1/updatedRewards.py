import optuna
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Define CustomBipedalReward so it is available for optimize_ppo
class CustomBipedalReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(reward, observation), terminated, truncated, info

    def reward(self, reward, observation):
        # This counters the negative penalties so "living" isn't a net loss.
        reward += 0.005

        # Hull Angle Penalty (Index 0)
        if (abs(observation[0]) > 0.5): reward -= 0.01 * abs(observation[0]) 

        # Penalty for high y velocity (Index 3)
        if (abs(observation[3]) > 0.5): reward -= 0.01 * abs(observation[3])

        # Reward for x velocity (Index 2)
        if observation[2] > 0: reward += 2.0 * observation[2]

        if observation[2] < 0: reward -= 1.0 * abs(observation[2])


        # Reward for joint movement (Index 5, 7, 9, 11)
        if abs(observation[0]) < 0.5:
            joint_speed_indices = [5, 7, 10, 12]
            for idx in joint_speed_indices:
                reward += 0.005 * abs(observation[idx])
        

        # Penalize "Kneeling" / Extreme Joint Angles
        joint_angle_indices = [4, 6, 9, 11]
        for idx in joint_angle_indices:
            if abs(observation[idx]) > 1.2:
                 reward -= 0.05

        return reward


def optimize_ppo(trial):
    # 1. Create base env
    env = gym.make("BipedalWalker-v3")
    
    # 2. Apply your custom reward
    env = CustomBipedalReward(env)
    
    # 3. Apply Vector Normalization (Crucial!)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # 4. Define hyperparameters from trial
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

    model = PPO("MlpPolicy", env, learning_rate=learning_rate, verbose=0)
    
    # Train for your 20,000 steps
    model.learn(total_timesteps=20000)


    # Create environment
    env = gym.make("BipedalWalker-v3")
    env = CustomBipedalReward(env)

    # Initialize agent with suggested hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        verbose=0
    )

    # Train the agent
    model.learn(total_timesteps=20000)

    # Evaluate the agent
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    return mean_reward


# Create the Optuna study
study = optuna.create_study(direction="maximize")

# Run the optimization
print("Starting optimization for 8 trials...")
study.optimize(optimize_ppo, n_trials=8)

# Print the best results
print("\nOptimization finished.")
print(f"Best Reward: {study.best_value}")
print("Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# With vecnormalize

# Create log directory
log_dir = "./PPO_bipedal_logs/"
os.makedirs(log_dir, exist_ok=True)

# 1. Recreate the EXACT environment pipeline used in Optuna
def make_env():
    env = gym.make("BipedalWalker-v3")
    env = CustomBipedalReward(env)
    env = Monitor(env, log_dir)
    return env

# Wrap for normalization
env = DummyVecEnv([make_env])
# Use norm_obs and norm_reward as you did in the study
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# 2. Initialize PPO with the best hyperparameters
best_params = study.best_params
model = PPO(
    "MlpPolicy",
    env,
    **best_params,
    verbose=1
)

# 3. Train for 100,000 timesteps
model.learn(total_timesteps=100000)

# --- VIDEO CAPTURE SECTION ---

# 4. Create a separate evaluation env that uses the TRAINING stats
render_env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
render_env = CustomBipedalReward(render_env) 
render_env = DummyVecEnv([lambda: render_env])

# Apply the normalization stats from the training env
render_env = VecNormalize(render_env, norm_obs=True, norm_reward=False, training=False)
render_env.obs_rms = env.obs_rms 

obs = render_env.reset()
frames = []

print("Capturing frames...")
for _ in range(1000): # Increased to 1000 for a longer video
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = render_env.step(action)

    # render_env is now a VecEnv, so we access the rgb_array differently
    frames.append(render_env.envs[0].render())

    if dones[0]:
        obs = render_env.reset()

render_env.close()

from IPython.display import HTML
from base64 import b64encode
import imageio

# Save the video
video_path = 'optuna_best_walker.mp4'
imageio.mimsave(video_path, frames, fps=30)

# Display the video
mp4 = open(video_path, 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML(f"""<video width=600 controls><source src="{data_url}" type="video/mp4"></video>""")