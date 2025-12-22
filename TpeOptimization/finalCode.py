import optuna
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os

# 1. Custom Reward Wrapper (Refined for Speed and Stability)
class CustomBipedalReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(reward, observation), terminated, truncated, info

    def reward(self, reward, observation):
        # Boost forward progress reward
        vel_x = observation[2]
        if vel_x > 0:
            reward += 1.5 * vel_x

        # Penalize excessive tilting
        reward -= 0.3 * abs(observation[0])

        return reward

# 2. Objective Function for Optuna
def objective(trial):
    # Setup environment
    def make_env():
        env = gym.make("BipedalWalker-v3")
        return CustomBipedalReward(env)

    env = DummyVecEnv([make_env])
    # Observation normalization is still crucial for SAC
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Bayesian Optimization Hyperparameter Space
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
    model.learn(total_timesteps=10000)

    # Evaluate using the normalized environment
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    env.close()
    return mean_reward

# 3. Run the Study with TPE Sampler
if __name__ == "__main__":
    # Specify the TPE Sampler here
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="SAC_Bipedal_Optimization"
    )

    print("Starting Bayesian Optimization with TPESampler...")
    study.optimize(objective, n_trials=5) # Increased trials to let BO learn the space

    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pickle
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common import results_plotter

# 1. Retrieve the best parameters
best_params = study.best_params
print(f"Retraining SAC with: {best_params}")

# Create log directory
log_dir = "./sac_bipedal_logs/"
os.makedirs(log_dir, exist_ok=True)

# 2. Setup Environment with Monitor
def make_final_env():
    env = gym.make("BipedalWalker-v3")
    env = CustomBipedalReward(env)
    # The Monitor wrapper logs episode rewards to the CSV file
    env = Monitor(env, log_dir)
    return env

env = DummyVecEnv([make_final_env])

# We wrap it in DummyVecEnv and VecNormalize just like in the study
train_env = DummyVecEnv([make_final_env])
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

# 6. Setup Video Environment
# Use render_mode="rgb_array" for frame capture
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

def plot_learning_curve(log_dir, title="Learning Curve"):
    """
    Plot the results using SB3's built-in tool and Matplotlib.
    """
    # Helper to load results and plot with a rolling window for smoothing
    plot_results([log_dir], 100000, results_plotter.X_TIMESTEPS, title)
    plt.show()

    # Manual plotting for more control (optional)
    results = load_results(log_dir)
    x, y = ts2xy(results, 'timesteps')

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, alpha=0.2, color='blue', label='Raw Reward')

    # Calculate moving average (window of 50 episodes)
    if len(y) > 50:
        y_smoothed = np.convolve(y, np.ones(50)/50, mode='valid')
        x_smoothed = x[len(x) - len(y_smoothed):]
        plt.plot(x_smoothed, y_smoothed, color='red', linewidth=2, label='Smoothed (MA 50)')

    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('SAC BipedalWalker Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_learning_curve(log_dir)