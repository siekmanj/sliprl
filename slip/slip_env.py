import mujoco_py as mj
import numpy as np
import os
import random

STOCHASTIC  = True
FRAMESKIP   = 10
ALIVE_BONUS = 1

class SlipEnv:
    def __init__(self):
        self.model = mj.load_model_from_path("./slip/slip.xml")
        self.sim = mj.MjSim(self.model, nsubsteps=FRAMESKIP)
        
        self.vis = mj.MjViewerBasic(self.sim)
        self.vis.cam.trackbodyid = 2
        self.vis.cam.distance = self.model.stat.extent * 0.75
        self.vis.cam.lookat[2] = 1.15
        self.vis.cam.elevation = -20

        self.desired_speed = random.choice([-2, 2])

        #obs_vec = np.concatenate([self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)])
        obs_vec = np.concatenate([self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)])
        obs_vec = np.append(obs_vec, self.desired_speed)
        self.observation_space = np.zeros(len(obs_vec))
        self.action_space = np.zeros(2)
        self.dt = self.model.opt.timestep * FRAMESKIP

        self.avg_spd = 0
        self.step_ctr = 1

    def reset(self):
        print("resetting environment. Average speed/desired speed:", self.avg_spd/self.step_ctr, " / ", self.desired_speed)
        self.desired_speed = random.choice([-2, 2])
        self.avg_spd = 0
        self.step_ctr = 1
        self.sim.reset()
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        if STOCHASTIC:
          qpos += np.random.uniform(low=-.005, high=.005, size=self.model.nq)
          qvel += np.random.uniform(low=-.005, high=.005, size=self.model.nv)

        old_state = self.sim.get_state()
        tmp = mj.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(tmp)
        self.sim.forward()

        #obs_vec = np.concatenate([self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)], [self.desired_speed])
        obs_vec = np.concatenate([self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)])
        obs_vec = np.append(obs_vec, self.desired_speed)

        return obs_vec

    def step(self, action):
        #if(len(action) != 3):
        #    print("SlipEnv: action dimension was not 2!")
        self.sim.data.ctrl[0] = 3000*action[0]
        self.sim.data.ctrl[1] = 4000*action[1]
        #self.sim.data.ctrl[2] = action[2]

        posbefore = self.sim.data.qpos[0]
        self.sim.step()
        posafter, height, ang = self.sim.data.qpos[0:3]
        velafter = self.sim.data.qvel[0]

        #reward = (posafter - posbefore) / self.dt
        #reward = -abs(self.desired_speed - (posafter - posbefore) / self.dt)

        reward  = 1.0
        reward -= abs(velafter - self.desired_speed)
        reward -= 1 * pow(self.sim.data.qvel[2], 2) 

        self.avg_spd += velafter
        self.step_ctr += 1
    
        #obs_vec = np.concatenate([self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)])
        #obs_vec = np.concatenate([self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)], [self.desired_speed])
        obs_vec = np.concatenate([self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)])
        obs_vec = np.append(obs_vec, self.desired_speed)
        done = not (np.isfinite(self.sim.data.qpos).all() and (self.sim.data.qpos[1] > 0.3) and (abs(self.sim.data.qpos[2]) < 2))
        if done:
          reward -= 20

        self.sim.data.ctrl[0] = 0
        self.sim.data.ctrl[1] = 0
        return obs_vec, reward, done, {}
        
    def render(self):
        self.vis.render()
