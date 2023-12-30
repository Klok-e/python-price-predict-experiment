import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from line_profiler.explicit_profiler import profile

from util import calculate_observation, OBS_PRICES_SEQUENCE, OBS_OTHER


class CustomEnv(gym.Env):
    def __init__(self, df_tickers, episode_length=1024,
                 commission=0.001, model_in_observations=64):
        super().__init__()

        self.commission = commission

        # 0: hold; 1: buy; 2: sell
        self.action_space = spaces.Discrete(3)

        prepro_dataset = df_tickers[0][0]
        pristine_dataset = df_tickers[0][1]
        self.FEATURES_SHAPE = calculate_observation(prepro_dataset, pristine_dataset, None, model_in_observations)[0]
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

        self.df_tickers = df_tickers
        self.episode_length = episode_length
        self.current_step = 0
        self.episode_start_index = 0
        self.episode_current_ticker = 0
        self.buy_price = None

        self.holdings = 0
        self.cash_balance = 1000

        self.model_in_observations = model_in_observations
        self.skip_steps = 1024 + self.model_in_observations

    @profile
    def step(self, action):
        # Update the step
        self.current_step += 1

        # Calculate the current observation and price information
        observation, curr_close, prev_close = self.calculate_observation()

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

    @profile
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.buy_price = None
        self.episode_current_ticker = random.randrange(0, len(self.df_tickers))
        self.episode_start_index = random.randint(self.skip_steps,
                                                  len(self.df_tickers[self.episode_current_ticker][
                                                          0]) - self.episode_length - 1)

        print(f"ticker {self.episode_current_ticker}; index {self.episode_start_index}")

        observation, _, _ = self.calculate_observation(self.current_step, self.episode_current_ticker)
        info = {}
        return observation, info

    def render(self):
        pass

    def close(self):
        pass

    @profile
    def calculate_observation(self):
        index_start = self.episode_start_index + self.current_step - self.skip_steps
        index_end = self.episode_start_index + self.current_step

        prepro_dataset = self.df_tickers[self.episode_current_ticker][0]
        pristine_dataset = self.df_tickers[self.episode_current_ticker][1]
        prepro_df = prepro_dataset.iloc[index_start:index_end]
        pristine_df = pristine_dataset.iloc[index_start:index_end]
        observation, curr_close, prev_close = calculate_observation(prepro_df, pristine_df, self.buy_price,
                                                                    self.model_in_observations)

        return observation, curr_close, prev_close
