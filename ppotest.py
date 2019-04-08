import random
import argparse

from rl.algos import PPO
from rl.envs.normalize import PreNormalizer
from rl.policies import GaussianMLP
from rl.utils import run_experiment

USE_HOPPER = False
USE_SLIP = not USE_HOPPER

if USE_HOPPER:
  from hopper.hopper_env import HopperEnv
elif USE_SLIP:
  from slip.slip_env import SlipEnv


def make_slip_env():
  def wrapper():
    if USE_HOPPER:
      return HopperEnv()
    elif USE_SLIP:
      return SlipEnv() 
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

  args.batch_size = 1500
  args.lr = 5e-3
  args.epochs = 5
  args.num_steps = 3000

  args.use_gae = False
  
  env = make_slip_env()
  
  obs_dim = env().observation_space.shape[0]
  action_dim = env().action_space.shape[0]

  policy = GaussianMLP(obs_dim, action_dim, nonlinearity='tanh', init_std=1, learn_std=True)
  normalizer = PreNormalizer(iter=100, noise_std=1, policy=policy, online=False)
  algo = PPO(args=vars(args))

  run_experiment(algo=algo, policy=policy, env_fn=env, args=args, normalizer=normalizer)

main()


