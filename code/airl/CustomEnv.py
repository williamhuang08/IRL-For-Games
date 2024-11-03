import gymnasium as gym
from gymnasium import spaces

"""
Define the environment
- STATES: the set of all possible observations for an agent
- ACTIONS: the set of all possible actions an agent can take
- STEP: determines how actions lead to changes in states (for CustomEnv, the new state is randomly sampled from uniform distribution)

In this case, the the states and actions are num_features-dimensional
""" 

class CustomEnv(gym.Env):
    def __init__(self, num_options: int = 2, weights=None):
        super().__init__()

        self.num_options = num_options
        self.observation_space = spaces.Box(low=-1, high=1, shape=(num_features,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_features,), dtype=np.float32)
        self.state = None
        self.max_steps = 100  # Define a fixed number of steps per episode
        self.current_step = 0

        # Initialize weights for reward calculation
        self.weights = weights if weights is not None else np.random.uniform(-1, 1, 10)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            np.random.seed(seed)
        self.state = self.observation_space.sample()
        self.current_step = 0  # Reset the step counter
        return self.state, {}

    def step(self, action):
        # Calculate the reward based on the weights
        reward = np.dot(self.state * action, self.weights)
        
        # Update the state
        self.state = np.random.uniform(low=-1, high=1, size=(num_features,)).astype(np.float32)
        
        # Increment step counter
        self.current_step += 1
        
        # Define the done condition based on max_steps
        done = self.current_step >= self.max_steps
        
        # Since done is used, no need for truncated in this context
        truncated = False  

        info = {
            "obs": self.state,
            "rews": reward,
        }

        return self.state, reward, done, truncated, info

    def render(self, mode='human'):
        pass  # Implement rendering logic as needed