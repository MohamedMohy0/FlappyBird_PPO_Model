import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


 

save_paths = {
    "PPO": "./flappy_best_ppo/best_model.zip",
}

env = gym.make("FlappyBird-v0", render_mode="human")


model = PPO.load(save_paths["PPO"], env=env)


n_episodes = 5
for ep in range(n_episodes):
    obs, info = env.reset()
    terminated, truncated = False, False
    total_reward = 0
    print(f"\n--- Episode {ep+1} ---")

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(f"Episode {ep+1} finished with reward: {total_reward}")

env.close()
