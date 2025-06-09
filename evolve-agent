import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random

# Simple MLP agent for Double Cartpole (continuous actions)
class SimpleMLP(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    def forward(self, x):
        return self.net(x)

def evaluate_agent(agent, env, episodes=3):
    total_reward = 0.0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state_v = torch.FloatTensor(state).unsqueeze(0)
            action = agent(state_v)
            # For Box action space: use tanh to bound to [-1, 1]
            action = torch.tanh(action).detach().cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            state = next_state
    return total_reward / episodes

def mutate_agent(agent, mutation_power=0.1):
    child = SimpleMLP(agent.net[0].in_features, agent.net[-1].out_features)
    child.load_state_dict({k: v.clone() for k, v in agent.state_dict().items()})
    with torch.no_grad():
        for param in child.parameters():
            param.add_(mutation_power * torch.randn_like(param))
    return child

def evolutionary_train(env_name='InvertedDoublePendulum-v4', pop_size=50, generations=100, elite_frac=0.2, mutation_power=0.1):
    # Curriculum parameters
    curriculum_steps = 1
    gravity_start, gravity_end = 9.8, 9.8  # Start easy, end at normal gravity
    friction_start, friction_end = .1, .1  # Start with high friction (easy), end at normal (hard)
    obs_size = gym.make(env_name).observation_space.shape[0]
    action_size = gym.make(env_name).action_space.shape[0]
    population = [SimpleMLP(obs_size, action_size) for _ in range(pop_size)]
    n_elite = max(1, int(pop_size * elite_frac))
    for curriculum_step in range(curriculum_steps):
        # Linearly interpolate gravity and friction
        gravity = gravity_start + (gravity_end - gravity_start) * (curriculum_step / (curriculum_steps - 1))
        friction = friction_start + (friction_end - friction_start) * (curriculum_step / (curriculum_steps - 1))
        print(f"\nCurriculum step {curriculum_step+1}/{curriculum_steps}: gravity={gravity:.2f}, friction={friction:.3f}")
        # Create environment with custom gravity/friction
        env = gym.make(env_name)
        # Set gravity and friction in MuJoCo env
        if hasattr(env.unwrapped, 'model'):
            env.unwrapped.model.opt.gravity[1] = -gravity
            # Friction is a vector, set all to desired value
            env.unwrapped.model.geom_friction[:] = friction
        gen = 1
        while True:
            fitness = [evaluate_agent(agent, env) for agent in population]
            elite_idxs = np.argsort(fitness)[-n_elite:]
            elites = [population[i] for i in elite_idxs]
            best_fitness = fitness[elite_idxs[-1]]
            best_agent = elites[-1]
            print(f"Curriculum {curriculum_step+1}, Gen {gen}: Best fitness = {best_fitness:.2f}")
            if best_fitness > 9300:
                print(f"Early stopping: fitness {best_fitness:.2f} > 9300")
                break
            # Reproduce
            new_population = elites[:]
            while len(new_population) < pop_size:
                parent = random.choice(elites)
                child = mutate_agent(parent, mutation_power)
                new_population.append(child)
            population = new_population
            gen += 1
        # After mastering the curriculum step, always display the best agent
        render_env = gym.make(env_name, render_mode='human')
        if hasattr(render_env.unwrapped, 'model'):
            render_env.unwrapped.model.opt.gravity[1] = -gravity
            render_env.unwrapped.model.geom_friction[:] = friction
        score = evaluate_agent(best_agent, render_env, episodes=1)
        print(f"[Curriculum Step Complete] Best agent score: {score:.2f}")
        render_env.close()
        env.close()
    # Final evaluation at normal gravity/friction
    env = gym.make(env_name)
    if hasattr(env.unwrapped, 'model'):
        env.unwrapped.model.opt.gravity[1] = -gravity_end
        env.unwrapped.model.geom_friction[:] = friction_end
    best_agent = elites[-1]
    final_score = evaluate_agent(best_agent, env, episodes=10)
    print(f"Final best agent average score (10 episodes): {final_score:.2f}")
    torch.save(best_agent.state_dict(), "best_double_cartpole_agent.pth")
    print("Agent completed all curriculum steps and was saved as best_double_cartpole_agent.pth.")

if __name__ == "__main__":
    evolutionary_train()
