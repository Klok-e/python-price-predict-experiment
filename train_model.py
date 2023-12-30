import random

import torch
from line_profiler import profile
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import CustomEnv
from model import LSTMExtractor
from util import download_and_process_data_if_available


@profile
def train_model(df_tickers, hidden_size: int, lstm_layers: int, net_arch: list[int], timesteps: int,
                model_window_size: int):
    random.seed(42)

    env = make_vec_env(CustomEnv, env_kwargs={"df_tickers": df_tickers,
                                              "model_in_observations": model_window_size},
                       n_envs=5, seed=42, vec_env_cls=SubprocVecEnv)

    policy_kvargs = dict(activation_fn=torch.nn.LeakyReLU,
                         features_extractor_class=LSTMExtractor,
                         features_extractor_kwargs=dict(lstm_hidden_size=hidden_size, lstm_layers=lstm_layers),
                         net_arch=net_arch)

    # {'gamma': 0.8, 'ent_coef': 0.02, 'gae_lambda': 0.92}
    rl_model = PPO("MultiInputPolicy", env,
                   verbose=1,
                   tensorboard_log="./tensorboard/",
                   ent_coef=0.02,
                   gae_lambda=0.92,
                   gamma=0.8,
                   policy_kwargs=policy_kvargs)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="rl-model/checkpoints/",
        name_prefix="rl_model",
        verbose=1,
        save_vecnormalize=True,
        save_replay_buffer=True
    )
    eval_env = make_vec_env(CustomEnv, env_kwargs={"df_tickers": df_tickers,
                                                   "episode_length": 2048,
                                                   "model_in_observations": model_window_size}, seed=42,
                            vec_env_cls=SubprocVecEnv)
    eval_callback = EvalCallback(eval_env, best_model_save_path="rl-model/best-model/best_model",
                                 log_path="rl-model/best-model/results", eval_freq=100_000, verbose=1,
                                 n_eval_episodes=5)

    rl_model.learn(total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback])


if __name__ == "__main__":
    df_tickers = download_and_process_data_if_available("cache/df_tickers.pkl")

    hidden_size_list = [128]
    lstm_layers_list = [1]
    arch_list = [[256, 256]]
    window_size_list = [64]
    for hidden_size in hidden_size_list:
        for lstm_layers in lstm_layers_list:
            for arch in arch_list:
                for window_size in window_size_list:
                    print(f"hidden_size {hidden_size}, lstm_layers {lstm_layers}, window_size {window_size}")
                    train_model(df_tickers, hidden_size, lstm_layers, arch, 1_000_000, window_size)
