import os
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from line_profiler.explicit_profiler import profile

from util import calculate_observation, OBS_PRICES_SEQUENCE, OBS_OTHER


class CustomEnv(gym.Env):
    def __init__(self, df_tickers, episode_length=1024, episodes_max=None,
                 commission=0.001, model_in_observations=64, random_gen=random.Random(os.getpid())):
        super().__init__()

        self.commission = commission

        # 0: hold; 1: buy; 2: sell
        self.action_space = spaces.Discrete(3)

        prepro_dataset = df_tickers[0][0]
        pristine_dataset = df_tickers[0][1]
        self.FEATURES_SHAPE = calculate_observation(
            prepro_dataset[:model_in_observations],
            pristine_dataset[:model_in_observations], None)[0]
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
        self.buy_price = None
        self.holdings = 0
        self.cash_balance = 1000

        self.model_in_observations = model_in_observations

        self.current_step = 0
        self.episode_idx = 0
        self.episodes = []
        for ticker in df_tickers:
            for start_index in range(0, len(ticker[0]), self.episode_length):
                self.episodes.append((ticker[0].iloc[start_index:start_index + self.episode_length],
                                      ticker[1].iloc[start_index:start_index + self.episode_length],
                                      ticker[2],
                                      ticker[3],))

        random_gen.shuffle(self.episodes)
        if episodes_max is not None:
            self.episodes = self.episodes[:episodes_max]

        assert len(self.episodes) > 0, "No episodes"

    @profile
    def step(self, action):
        # Calculate the current observation and price information
        observation, curr_close, prev_close = self.calculate_observation()

        # Initialize reward and transaction cost
        transaction_cost = 0

        # Define the actions
        if action == 1 and self.buy_price is None:  # Buy
            trade_vector = 1  # Buying 1 unit
            self.holdings += trade_vector
            transaction_cost = self.commission * curr_close * trade_vector
            self.cash_balance -= (curr_close * trade_vector + transaction_cost)
            self.buy_price = curr_close
        elif action == 2 and self.buy_price is not None:  # Sell
            trade_vector = -1  # Selling 1 unit
            self.holdings += trade_vector
            transaction_cost = self.commission * curr_close * abs(trade_vector)
            self.cash_balance += (curr_close * abs(trade_vector) - transaction_cost)
            self.buy_price = None  # Reset buy_price upon sale

        assert self.holdings <= 1

        # Calculate the reward using the formula provided
        portfolio_value_t = self.cash_balance + np.dot(prev_close, self.holdings)
        portfolio_value_t_plus_1 = self.cash_balance + np.dot(curr_close, self.holdings)
        reward = (portfolio_value_t_plus_1 - portfolio_value_t - transaction_cost) / portfolio_value_t

        reward = np.clip(reward, -1, 1)

        # Update the step
        self.current_step += 1

        # Check if the episode is terminated
        terminated = len(self.episodes[self.episode_idx][0]) < self.current_step + self.model_in_observations

        # Return the step information
        return observation, reward, terminated, False, {}

    @profile
    def reset(self, seed=None, options=None):
        self.episode_idx += 1
        if self.episode_idx > len(self.episodes) - 1:
            self.episode_idx = 0
        self.current_step = 0

        self.buy_price = None
        self.holdings = 0
        self.cash_balance = 1000

        # print(f"episode {self.episode_idx}; ticker {self.episodes[self.episode_idx][3]}")

        observation, _, _ = self.calculate_observation()
        info = {}
        return observation, info

    def render(self):
        pass

    def close(self):
        pass

    @profile
    def calculate_observation(self):
        prepro_dataset = self.episodes[self.episode_idx][0][
                         self.current_step:self.current_step + self.model_in_observations]
        pristine_dataset = self.episodes[self.episode_idx][1][
                           self.current_step:self.current_step + self.model_in_observations]
        observation, curr_close, prev_close = calculate_observation(prepro_dataset, pristine_dataset, self.buy_price)

        return observation, curr_close, prev_close
