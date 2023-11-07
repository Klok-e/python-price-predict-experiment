import random
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, dataset: pd.DataFrame, scaler: MultiScaler, episode_length=1024,
                 commission=0.001):
        super().__init__()

        self.commission = commission

        # 0: hold; 1: buy; 2: sell
        self.action_space = spaces.Discrete(3)

        self.FEATURES_SHAPE = calculate_observation(dataset, scaler, None)[0]
        self.FEATURES_SHAPE = {k: v.shape for k, v in self.FEATURES_SHAPE.items()}
        print(f"obs length = {self.FEATURES_SHAPE}")

        # Update the observation space to include extra information
        self.observation_space = spaces.Dict(
            {OBS_PRICES_SEQUENCE: spaces.Box(low=-1, high=1,
                                             shape=self.FEATURES_SHAPE[OBS_PRICES_SEQUENCE],
                                             dtype=np.float32),
             OBS_OTHER: spaces.Box(low=-1, high=1,
                                   shape=self.FEATURES_SHAPE[OBS_OTHER],
                                   dtype=np.float32)})

        self.dataset = dataset
        self.scaler = scaler
        self.episode_length = episode_length
        self.current_step = 0
        self.start_index = 0
        self.buy_price = None

        self.holdings = 0
        self.cash_balance = 1000

    def step(self, action):
        # Update the step
        self.current_step += 1

        # Calculate the current observation and price information
        observation, curr_close, prev_close = self.calculate_observation(self.current_step)

        # Initialize reward and transaction cost
        reward = 0
        transaction_cost = 0

        # Define the actions
        if action == 1 and self.buy_price is None:  # Buy
            trade_vector = 1  # Buying 1 unit
            self.buy_price = curr_close
            self.holdings += trade_vector
            transaction_cost = self.commission * curr_close * trade_vector
            self.cash_balance -= (curr_close * trade_vector + transaction_cost)
        elif action == 2 and self.buy_price is not None:  # Sell
            trade_vector = -1  # Selling 1 unit
            self.holdings += trade_vector
            transaction_cost = self.commission * curr_close * abs(trade_vector)
            self.cash_balance += (curr_close * abs(trade_vector) - transaction_cost)
            self.buy_price = None  # Reset buy_price upon sale

        # Calculate the reward using the formula provided
        portfolio_value_t = self.cash_balance + np.dot(prev_close, self.holdings)
        portfolio_value_t_plus_1 = self.cash_balance + np.dot(curr_close, self.holdings)
        reward = (portfolio_value_t_plus_1 - portfolio_value_t) - transaction_cost

        # Check if the episode is terminated
        terminated = self.current_step >= self.episode_length

        # Return the step information
        return observation, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.buy_price = None
        self.start_index = random.randint(SKIP_STEPS, len(self.dataset) - self.episode_length - 1)
        observation, _, _ = self.calculate_observation(self.current_step)
        info = {}
        return observation, info

    def render(self):
        pass

    def close(self):
        pass

    def calculate_observation(self, current_step):
        index_start = self.start_index + current_step - SKIP_STEPS
        index_end = self.start_index + current_step

        df = self.dataset.iloc[index_start:index_end]
        observation, curr_close, prev_close = calculate_observation(df, self.scaler, self.buy_price)

        return observation, curr_close, prev_close