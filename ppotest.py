import random
import argparse

from rl.algos import PPO
from rl.envs.normalize import PreNormalizer
from rl.policies import GaussianMLP
from rl.utils import run_experiment

from slip.slip_env import SlipEnv


def make_slip_env():
  def wrapper():
    return SlipEnv() #for some reason, need to wrap env in a function for run_experiment
  return wrapper

def main():
  random.seed(1)
  
  parser = argparse.ArgumentParser()

  PPO.add_arguments(parser)

  parser.add_argument("--seed", type=int, default=1,
                      help="RNG seed")
  parser.add_argument("--logdir", type=str, default="./LOG/",
                      help="Where to log diagnostics to")
  parser.add_argument("--name", type=str, default="model")

  args = parser.parse_args()

  args.batch_size = 900
  args.lr = 2.5e-4
  args.epochs = 5
  args.num_steps = 1000

  args.use_gae = False
  
  env = make_slip_env()
  
  obs_dim = env().observation_space.shape[0]
  action_dim = env().action_space.shape[0]

  policy = GaussianMLP(obs_dim, action_dim, nonlinearity='tanh', init_std=0.15, learn_std=False)
  normalizer = PreNormalizer(iter=100, noise_std=1, policy=policy, online=False)
  algo = PPO(args=vars(args))

  run_experiment(algo=algo, policy=policy, env_fn=env, args=args, normalizer=normalizer)

main()


