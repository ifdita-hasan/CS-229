from q_network import DQLAgent, ReplayBuffer, Q_network

# Sample code that trains on gym env using 

# import gym
# import torch
# import numpy as np


# # Set hyperparameters
# state_size = 4  # size of EACH state
# action_size = 2 # size of EACH action 
# buffer_size = int(1e5)
# batch_size = 64
# gamma = 0.99
# lr = 5e-4
# tau = 1e-3
# update_every = 4

# # Create environment, agent, and replay buffer
# env = gym.make('CartPole-v0')
# agent = DQLAgent(state_size, action_size, seed=0, lr=lr, gamma=gamma, tau=tau, update_every=update_every)
# buffer = ReplayBuffer(buffer_size, batch_size, seed=0)

# # Train the agent
# n_episodes = 1000
# max_timesteps = 1000
# eps_start = 1.0
# eps_end = 0.01
# eps_decay = 0.995

# eps = eps_start
# for i_episode in range(1, n_episodes+1):
#     state = env.reset()
#     score = 0
#     for t in range(max_timesteps):
#         action = agent.act(state, eps)
#         next_state, reward, done, _ = env.step(action)
#         buffer.add(state, action, reward, next_state, done)
#         if len(buffer) > batch_size:
#             experiences = buffer.sample()
#             agent.train(experiences, batch_size)
#         score += reward
#         state = next_state
#         if done:
#             break
#     eps = max(eps_end, eps_decay*eps)
#     print(f'Episode {i_episode} - Score: {score}')

# env.close()

# Define the game environment and necessary parameters
state_size = ...
action_size = ...
hidden_size = ...
learning_rate = ...
gamma = ...
epsilon_start = ...
epsilon_end = ... 
epsilon_decay = ... 
buffer_size = ...   # maximium size of replay buffer
batch_size = ...    # size of each minibach 
min_epsilon = 0.01 
seed = ...

# Set the maximum number of episodes and maximum number of steps per episode
max_episodes = ...
max_steps = ...


# Create an instance of the DQNAgent class
agent = DQLAgent(state_size, action_size, hidden_size, batch_size, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay)

# Create an instance of the ReplayBuffer class
memory = ReplayBuffer(buffer_size, batch_size, seed)

# Train the agent
for episode in range(max_episodes):
    # Reset the environment and get the initial state
    state = ...

    for step in range(max_steps):
        # Select an action using the DQNAgent
        action = agent.act(state)

        # Execute the action and observe the next state, reward, and done flag
        next_state, reward, end_state = ...

        # Add the experience tuple to the replay buffer
        memory.add(state, action, reward, next_state, end_state)

        # Sample a batch of experience tuples from the replay buffer
        if len(memory) < batch_size: 
            continue
        
        batch = memory.sample(batch_size)

        # Train the Q-network using the sampled batch of experience tuples
        agent.train(batch)

        # Update the exploration parameter epsilon
        agent.update_epsilon(episode, min_epsilon)

        if end_state:
            break

        # Set the state for the next step
        state = next_state
