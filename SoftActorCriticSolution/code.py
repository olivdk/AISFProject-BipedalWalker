import optuna
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pickle

# 1. Reward Wrapper 
class CustomBipedalReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(reward, observation), terminated, truncated, info

    def reward(self, reward, observation):
        # Forward progress reward
        vel_x = observation[2]
        if vel_x > 0:
            reward += 1.5 * vel_x

        # Penalize excessive tilting
        reward -= 0.3 * abs(observation[0])

        return reward

# 2. Objective Function
def objective(trial):
    # Setup environment
    def make_env():
        env = gym.make("BipedalWalker-v3")
        return CustomBipedalReward(env)

    env = DummyVecEnv([make_env])
    # Observation normalization is still crucial for SAC
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Optimization Hyperparameter Space
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [100000, 300000]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "tau": trial.suggest_float("tau", 0.005, 0.02),
        "gamma": trial.suggest_float("gamma", 0.98, 0.999),
        "learning_starts": 2000, # Initial random exploration
    }

    # Initialize SAC
    model = SAC(
        "MlpPolicy",
        env,
        **params,
        verbose=0
    )

    # Train for a shorter period during study
    model.learn(total_timesteps=20000)

    # Evaluate using the normalized environment
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    env.close()
    return mean_reward

# 3. Study
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    print("Starting Bayesian Optimization...")
    study.optimize(objective, n_trials=5)

    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

# 1. Retrieve the best parameters
best_params = study.best_params
print(f"Retraining SAC with: {best_params}")

# 2. Setup the training environment
def make_env():
    env = gym.make("BipedalWalker-v3")
    return CustomBipedalReward(env)

# We wrap it in DummyVecEnv and VecNormalize just like in the study
train_env = DummyVecEnv([make_env])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

# 3. Initialize the Final Model
# 'ent_coef="auto"' enables automatic entropy tuning (highly recommended for SAC)
model = SAC(
    "MlpPolicy",
    train_env,
    **best_params,
    ent_coef="auto", 
    verbose=1,
    tensorboard_log="./sac_bipedal_logs/"
)

# 4. Long-term training
print("Starting 100,000 step training...")
model.learn(total_timesteps=100000)

# 5. Save the model and the normalization stats
model.save("sac_bipedal_walker_final")
train_env.save("vec_normalize_sac.pkl")

# --- EVALUATION AND VIDEO ---

# 6. Setup Video Environment
eval_env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
eval_env = CustomBipedalReward(eval_env)
eval_env = DummyVecEnv([lambda: eval_env])

# Load the saved normalization statistics into the eval environment
eval_env = VecNormalize.load("vec_normalize_sac.pkl", eval_env)
eval_env.training = False  # Don't update stats during testing
eval_env.norm_reward = False # We want raw rewards for final score check

# 7. Capture Frames
frames = []
obs = eval_env.reset()
print("Capturing video frames...")

for _ in range(1000): # Capture ~20 seconds of walking
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    
    # Grab frame from the underlying gym env
    frames.append(eval_env.envs[0].render())
    
    if done:
        obs = eval_env.reset()

print(f"Captured {len(frames)} frames. Ready for video export.")
eval_env.close()

from IPython.display import HTML
from base64 import b64encode
import imageio

# Ensure 'frames' list is available from the video capture cell (kfEqE7VhVUSW)
if 'frames' in locals() and frames:
    # Save the video
    video_path = 'sac_bipedal_walker_simulation.mp4'
    imageio.mimsave(video_path, frames, fps=30)

    # Display the video
    mp4 = open(video_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display(HTML(f"""<video width=600 controls><source src="{data_url}" type="video/mp4"></video>"""))
else:
    print("Video frames not available. Please run the model training and video capture cell first.")