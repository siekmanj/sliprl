from slip_env import SlipEnv
import time
import random
env = SlipEnv()

while True:
	action = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
	env.step(action)
	env.render()
	env.compute_reward()
	time.sleep(1)
