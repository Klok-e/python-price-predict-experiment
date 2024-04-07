import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import CustomEnv
from model import LSTMExtractor, MLPExtractor
from util import download_and_process_data_if_available

count = 0


# @profile
def train_model(df_tickers, net_arch: list[int], timesteps: int,
                model_window_size: int, n_envs: int, directory: str, model_save_name, policy_kwargs: dict):
    split_date = "2024-03-01"

    df_tickers_train = list(
        map(lambda ticker: (ticker[0].loc[:split_date], ticker[1].loc[:split_date], ticker[2], ticker[3]), df_tickers))
    df_tickers_test = list(
        map(lambda ticker: (ticker[0].loc[split_date:], ticker[1].loc[split_date:], ticker[2], ticker[3]), df_tickers))

    env = make_vec_env(CustomEnv, env_kwargs={"df_tickers": df_tickers_train,
                                              "model_in_observations": model_window_size},
                       n_envs=n_envs, seed=42, vec_env_cls=SubprocVecEnv)

    policy_kvargs = dict(activation_fn=torch.nn.LeakyReLU,
                         net_arch=net_arch,
                         **policy_kwargs)

    # {'gamma': 0.8, 'ent_coef': 0.02, 'gae_lambda': 0.92}
    rl_model = PPO("MultiInputPolicy", env,
                   verbose=1,
                   tensorboard_log=f"{directory}/tensorboard/",
                   ent_coef=0.02,
                   gae_lambda=0.92,
                   gamma=0.9,
                   policy_kwargs=policy_kvargs,
                   batch_size=1024,
                   seed=42)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=f"{directory}/rl-model/{model_save_name}/checkpoints/",
        name_prefix=model_save_name,
        verbose=1,
        save_vecnormalize=True,
        save_replay_buffer=True
    )
    eval_env = make_vec_env(CustomEnv, env_kwargs={"df_tickers": df_tickers_test,
                                                   "model_in_observations": model_window_size,
                                                   "episodes_max": 5},
                            seed=42,
                            vec_env_cls=SubprocVecEnv)
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=f"{directory}/rl-model/{model_save_name}/best-model",
                                 log_path=f"{directory}/rl-model/{model_save_name}/best-model/results",
                                 eval_freq=max(100_000 // n_envs, 1), verbose=1,
                                 n_eval_episodes=5)

    learn = rl_model.learn(total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback],
                           tb_log_name=model_save_name)

    env.unwrapped.close()
    eval_env.unwrapped.close()

    return learn


if __name__ == "__main__":
    df_tickers = download_and_process_data_if_available("cache/df_tickers.pkl")

    env = CustomEnv(df_tickers)
    check_env(env)
    del env

    hidden_size_list = [128, 256, 512]
    lstm_layers_list = [1, 2]
    arch_list = [[256, 256, 256]]
    window_size_list = [32, 64, 128]
    for hidden_size in hidden_size_list:
        for lstm_layers in lstm_layers_list:
            for arch in arch_list:
                for window_size in window_size_list:
                    print(f"hidden_size {hidden_size}, lstm_layers {lstm_layers}, window_size {window_size}")
                    train_model(df_tickers, arch, 500_000, window_size, 5, "data",
                                f"hs{hidden_size}_lstm{lstm_layers}_net{arch}_ws{window_size}",
                                dict(features_extractor_class=LSTMExtractor,
                                     features_extractor_kwargs=dict(lstm_hidden_size=hidden_size,
                                                                    lstm_layers=lstm_layers)))
