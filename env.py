import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from util import (
    calculate_observation,
    OBS_PRICES_SEQUENCE,
    OBS_OTHER,
    SharedPandasDataFrame,
)


class CustomEnv(gym.Env):
    def __init__(
        self,
        df_tickers,
        episode_length=1024,
        episodes_max=None,
        commission=0.001,
        model_in_observations=64,
    ):
        super().__init__()

        self.random_gen = None

        self.commission = commission

        # 0: hold; 1: buy; 2: sell
        self.action_space = spaces.Discrete(3)

        try:
            prepro_dataset = df_tickers[0][0].read()
            pristine_dataset = df_tickers[0][1].read()
        except:
            prepro_dataset = df_tickers[0][0]
            pristine_dataset = df_tickers[0][1]
        self.FEATURES_SHAPE = calculate_observation(
            prepro_dataset[:model_in_observations],
            pristine_dataset[:model_in_observations],
            None,
        )[0]
        self.FEATURES_SHAPE = {k: v.shape for k, v in self.FEATURES_SHAPE.items()}
        print(f"obs length = {self.FEATURES_SHAPE}")

        # Update the observation space to include extra information
        self.observation_space = spaces.Dict(
            {
                OBS_PRICES_SEQUENCE: spaces.Box(
                    low=-1,
                    high=1,
                    shape=self.FEATURES_SHAPE[OBS_PRICES_SEQUENCE],
                    dtype=np.float32,
                ),
                OBS_OTHER: spaces.Box(
                    low=-1,
                    high=1,
                    shape=self.FEATURES_SHAPE[OBS_OTHER],
                    dtype=np.float32,
                ),
            }
        )

        self.df_tickers = df_tickers
        self.episode_length = episode_length
        self.buy_price = None
        self.holdings = 0
        self.cash_balance = 10000
        self.operations_performed = 0

        self.model_in_observations = model_in_observations

        self.current_step = 0
        self.episode_idx = 0
        self.episodes = []
        for ticker in df_tickers:
            try:
                ticker_0 = ticker[0].read()
                ticker_1 = ticker[1].read()
            except:
                ticker_0 = ticker[0]
                ticker_1 = ticker[1]
            for start_index in range(0, len(ticker_0), self.episode_length):
                if (
                    len(ticker_0.iloc[start_index : start_index + self.episode_length])
                    == self.episode_length
                    and len(
                        ticker_1.iloc[start_index : start_index + self.episode_length]
                    )
                    == self.episode_length
                ):
                    self.episodes.append(
                        (
                            ticker_0.iloc[
                                start_index : start_index + self.episode_length
                            ],
                            ticker_1.iloc[
                                start_index : start_index + self.episode_length
                            ],
                            ticker[2],
                            ticker[3],
                        )
                    )

        self.episodes_max = episodes_max

    # @profile
    def step(self, action):
        # Calculate the current observation and price information
        observation, curr_close, prev_close = self.calculate_observation()

        # Initialize reward and transaction cost
        transaction_cost = 0

        # Define the actions
        if action == 1 and self.buy_price is None:  # Buy
            # use all available cash
            trade_vector = self.cash_balance / curr_close

            self.holdings += trade_vector
            transaction_cost = self.commission * curr_close * trade_vector
            self.cash_balance -= curr_close * trade_vector + transaction_cost
            self.buy_price = curr_close
            self.operations_performed += 1
        elif action == 2 and self.buy_price is not None:  # Sell
            trade_vector = -self.holdings  # Selling all units

            self.holdings += trade_vector
            transaction_cost = self.commission * curr_close * abs(trade_vector)
            self.cash_balance += curr_close * abs(trade_vector) - transaction_cost
            self.buy_price = None  # Reset buy_price upon sale
            self.operations_performed += 1

        assert self.holdings >= 0

        # Calculate the reward using the formula provided
        portfolio_value_t = self.cash_balance + np.dot(prev_close, self.holdings)
        portfolio_value_t_plus_1 = self.cash_balance + np.dot(curr_close, self.holdings)
        reward = (
            portfolio_value_t_plus_1 - portfolio_value_t - transaction_cost
        ) / portfolio_value_t

        # print(
        #     f"portfolio_value_t_plus_1 {portfolio_value_t_plus_1};"
        #     + f" portfolio_value_t {portfolio_value_t};"
        #     + f" transaction_cost {transaction_cost};"
        #     + f" reward {reward};"
        # )

        # if self.operations_performed < 2:
        #     reward = -1.0

        reward = np.clip(reward, -1.0, 1.0)

        # Update the step
        self.current_step += 1

        # Check if the episode is terminated
        terminated = False
        if (
            len(self.episodes[self.episode_idx][0])
            < self.current_step + self.model_in_observations
            or portfolio_value_t_plus_1 <= 10
        ):
            terminated = True

        # Return the step information
        return observation, reward, terminated, False, {}

    # @profile
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.random_gen = random.Random(seed)
            self.random_gen.shuffle(self.episodes)
            print(f"Initialized with seed {seed}")

            if self.episodes_max is not None:
                self.episodes = self.episodes[: self.episodes_max]
                print("Max episodes cut")

            assert len(self.episodes) > 0, "No episodes"

        self.episode_idx += 1
        if self.episode_idx > len(self.episodes) - 1:
            self.random_gen.shuffle(self.episodes)
            print(f"All {self.episode_idx} episodes finished")
            self.episode_idx = 0

        self.current_step = 0

        self.buy_price = None
        self.holdings = 0
        self.cash_balance = 10000
        self.operations_performed = 0

        # print(f"episode {self.episode_idx}; ticker {self.episodes[self.episode_idx][3]}")

        observation, _, _ = self.calculate_observation()
        info = {}
        return observation, info

    def render(self):
        pass

    def close(self):
        pass

    # @profile
    def calculate_observation(self):
        prepro_dataset = self.episodes[self.episode_idx][0][
            self.current_step : self.current_step + self.model_in_observations
        ]
        pristine_dataset = self.episodes[self.episode_idx][1][
            self.current_step : self.current_step + self.model_in_observations
        ]
        observation, curr_close, prev_close = calculate_observation(
            prepro_dataset, pristine_dataset, self.buy_price
        )

        return observation, curr_close, prev_close
