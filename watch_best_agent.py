import gymnasium as gym
import torch
import numpy as np
from evolve_cartpole import SimpleMLP

def watch_best_agent(env_name='InvertedDoublePendulum-v4', model_path='best_double_cartpole_agent.pth', episodes=5, gravity=9.8, friction=0.1):
    env = gym.make(env_name, render_mode='human')
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = SimpleMLP(obs_size, action_size)
    agent.load_state_dict(torch.load(model_path, map_location='cpu'))
    agent.eval()
    if hasattr(env.unwrapped, 'model'):
        env.unwrapped.model.opt.gravity[1] = -gravity
        env.unwrapped.model.geom_friction[:] = friction
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            state_v = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = agent(state_v)
                action = torch.tanh(action).cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            state = next_state
        print(f"Episode {ep+1}: reward = {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    watch_best_agent()
