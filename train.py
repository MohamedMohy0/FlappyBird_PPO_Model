import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import  PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

train_env = make_vec_env("FlappyBird-v0", n_envs=4)
eval_env = gym.make("FlappyBird-v0") 


ppo_save = "./flappy_best_ppo/"


eval_callback_ppo = EvalCallback(
    eval_env,
    best_model_save_path=ppo_save,
    log_path="./logs_ppo/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

ppo_policy_kwargs = dict(net_arch=[256, 256])

ppo_model = PPO(
    "MlpPolicy",
    train_env,
    policy_kwargs=ppo_policy_kwargs,
    verbose=1,
    learning_rate=0.0003,
    gamma=0.99,
    n_steps=1024,         
    batch_size=256,       
    gae_lambda=0.95,
    ent_coef=0.01,
    n_epochs=10,
    clip_range=0.2,
    tensorboard_log="./tb_ppo/"
)

print("🚀 Training PPO...")
ppo_model.learn(total_timesteps=1_000_000, callback=eval_callback_ppo)

def test_model(env, model, episodes=5):
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()
        print(f"Episode {ep+1}: Reward = {total_reward}")
    env.close()



print("Best Model PPO")
best_ppo = PPO.load(ppo_save + "best_model.zip")
test_model(eval_env, best_ppo)
