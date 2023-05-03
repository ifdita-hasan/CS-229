import torch
import torch.nn as nn 
import pprint
import numpy as np
import random
from collections import deque, namedtuple


pp = pprint.PrettyPrinter()


# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILON = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100


# dimensions of feature map for each training example in our model 
NUM_STATES = 100 # Putting in a random value for now 
NUM_ACTIONS = 100 # Putting in a random value for now 
# env = gym.make("CartPole-v0")
# env = env.unwrapped
# NUM_ACTIONS = env.action_space.n
# NUM_STATES = env.observation_space.shape[0]
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape





class Q_network(nn.Module):
    '''
    Q_network will take in a state and return Q_values of all possible actions. 
    Input = array of size state_dim.
    Output = array of size action_dim. 
    '''
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Q_network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class DQLAgent:
    '''
    DQL agent will explore using epsilon-greedy policy and then train the Q_network

    '''

    def __init__(self, state_size, action_size, hidden_size, batch_size, lr, gamma, eps_start, eps_end, eps_decay):

        # Store variables for the class
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.loss_fn = nn.MSELoss()

        # Define the two Q-networks to be used: 
        self.q_network = Q_network(state_size, action_size, hidden_size)
        self.target_q_network = Q_network(state_size, action_size, hidden_size)
        
        # Initialise both of them to have the same weights 
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Disable dropout for target_q_network which is only used for evaluation 
        self.target_q_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

        # replay buffer: 
        self.memory = []
        
        # epsilon for epsilon-greedy algorithm to do exploration
        self.epsilon = self.eps_start

    def act(self, current_state):
        # use epsilon-greedy algorithm to do exploration
        
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():  # We don't need gradients right now so do not need to build computation graphs 
                # q_network expects a batch of inputs. 
                state = torch.from_numpy(state).float().unsqueeze(0)
                q_values = self.q_network(state)
                # choose action with the highest Q_value in this state 
                action = q_values.argmax().item()
        return action


    def train(self, memory, batch_size):
        '''
        Train the Q_network using vanilla DQL algorithm
        '''
        # need at least batch_size of tuples 
        if len(memory) < batch_size:
            return
        
        # Sample a batch of transitions from memory
        batch = memory.sample(batch_size)
        states, actions, rewards, next_states, end_state = zip(*batch)

        # vertically stack the components so that each tensor has shape (batch_size, _) where _ could be state_dim, action_dim, or 1 (for rewards)
        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        end_state = torch.from_numpy(np.vstack(end_state).astype(np.uint8)).float()

        # Compute Q-values for current states and next states
        
        # Q_values of each state-action pair. 
        current_q_values = self.q_network(states).gather(1, actions) # tensor will have shape (batch_size, 1)

        # Q_value of next state and the best possible action from the next state. 
        next_q_values = self.q_network(next_states).max(1)[0].unsqueeze(1) # tensor will have shape (batch_size, 1)
                                                                           
        target_q_values = rewards + (self.gamma * next_q_values * (1 - end_state)) # tensor will have shape (batch_size, 1)

        # Compute the loss and update the Q-network
        loss = self.loss_fn(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self, episode, min_epsilon):
        self.epsilon = max(min_epsilon, self.epsilon * (1 - episode / 200))
    

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "end_state"])

    def add(self, state, action, reward, next_state, end_state):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, end_state)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)