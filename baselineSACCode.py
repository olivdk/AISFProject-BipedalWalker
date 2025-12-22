import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common import results_plotter

# Create a separate log directory for the baseline to avoid overwriting your optimized results
log_dir_baseline = "./sac_baseline_logs/"
os.makedirs(log_dir_baseline, exist_ok=True)

# 1. Setup Environment (Vanilla/Default)
def make_baseline_env():
    # Use the default environment with no custom reward wrappers
    env = gym.make("BipedalWalker-v3")
    # Monitor still used to record rewards for the final graph
    env = Monitor(env, log_dir_baseline)
    return env

# Wrap in DummyVecEnv as is standard for SB3, but NO VecNormalize
train_env = DummyVecEnv([make_baseline_env])

# 2. Initialize the Baseline Model
model = SAC(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log=log_dir_baseline
)

# 3. Long-term training
print("Starting 100,000 step Baseline training...")
model.learn(total_timesteps=100000)

# 4. Save the baseline model
model.save("sac_bipedal_baseline")

# --- EVALUATION AND VIDEO ---

# 5. Setup Video Environment (Vanilla)
eval_env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
# We do not use the custom reward wrapper or VecNormalize here
eval_env = DummyVecEnv([lambda: eval_env])

# 6. Capture Frames
frames = []
obs = eval_env.reset()
print("Capturing baseline video frames...")

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)

    # Grab frame from the underlying gym env
    frames.append(eval_env.envs[0].render())

    if done:
        obs = eval_env.reset()

print(f"Captured {len(frames)} frames for baseline.")
eval_env.close()

from IPython.display import HTML
from base64 import b64encode
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

# --- PART 1: VIDEO GENERATION ---
# This will look for the 'frames' list generated in your previous cell
if 'frames' in locals() and frames:
    video_path = 'sac_bipedal_walker_simulation.mp4'
    # Use macroblock_size=1 to avoid resolution mismatch errors in some environments
    imageio.mimsave(video_path, [np.array(f) for f in frames], fps=30)

    mp4 = open(video_path, 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    print(f"Displaying video: {video_path}")
    display(HTML(f"""<video width=600 controls><source src="{data_url}" type="video/mp4"></video>"""))
else:
    print("Video frames not found. Check if the capture cell ran successfully.")

# --- PART 2: GRAPH GENERATION ---
def compare_results(log_dirs, labels, title="SAC BipedalWalker: Baseline vs Optimized"):
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'green', 'red']

    for idx, log_folder in enumerate(log_dirs):
        if not os.path.exists(os.path.join(log_folder, "monitor.csv")):
            print(f"No logs found in {log_folder}, skipping...")
            continue

        results = load_results(log_folder)
        x, y = ts2xy(results, 'timesteps')

        # Plot raw data with high transparency
        plt.plot(x, y, alpha=0.1, color=colors[idx])

        # Calculate moving average for smooth visualization
        if len(y) > 50:
            window = 50
            y_smoothed = np.convolve(y, np.ones(window)/window, mode='valid')
            x_smoothed = x[len(x) - len(y_smoothed):]
            plt.plot(x_smoothed, y_smoothed, color=colors[idx], linewidth=2, label=labels[idx])

    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- PART 3: EXECUTION ---
log_paths = ["./sac_baseline_logs/", "./sac_bipedal_logs/"]
labels = ["Baseline (Default SAC)", "Fully Optimized"]

compare_results(log_paths, labels)