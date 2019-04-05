import mujoco_py as mj
import numpy as np
import os
import random

class SlipEnv:
	def __init__(self):
		model = mj.load_model_from_path("./slip/slip.xml")
		self.sim = mj.MjSim(model, nsubsteps=10)
		self.vis = mj.MjViewerBasic(self.sim)
		obs_vec = np.concatenate((self.sim.data.qpos, self.sim.data.qvel))
		self.observation_space = np.zeros(len(obs_vec))
		self.action_space = np.zeros(2)
		self.desired_speed = 1.0

	def reset(self):
		self.sim.reset()
		self.desired_speed = 1.0

		obs_vec = np.concatenate((self.sim.data.qpos, self.sim.data.qvel))
		return obs_vec

	def step(self, action):
		if(len(action) != 2):
			print("SlipEnv: action dimension was not 2!")
		self.sim.data.ctrl[0] = action[0]
		self.sim.data.ctrl[1] = action[1]
		self.sim.step()
	
		obs_vec = np.concatenate((self.sim.data.qpos, self.sim.data.qvel))
		reward = self.compute_reward()
		done = self.sim.data.qpos[1] < 0.2
		return obs_vec, reward, done, {}
		
	def render(self):
		self.vis.render()

	def compute_reward(self):
		vel_score = abs(self.desired_speed - self.sim.data.qvel[0])
		if self.sim.data.qpos[1] < 0.2:
			return -100
		return 1 - vel_score
