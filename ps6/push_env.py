"""Drake observation wrapper and episode summaries for hybrid push actions."""
import argparse

import json

from pathlib import Path

import time

import numpy as np

import torch

import gymnasium as gym

from gymnasium import spaces

from multi_push import MultiPush

ROOT = Path(__file__).parent/'results/multi_push_learning'

LENGTHS = np.array([.045, .085, .125])

FACES = [(0,1), (0,-1), (1,1), (1,-1)]
OBSERVATION_VERSION = 'resting_configuration_v1'
OBSERVATION_DIM = 13


def configuration_observations(observations):
    """Project archived 22-channel BC examples onto the 13 configuration inputs.

    This drops the countdown and all eight velocities, without injecting
    fixed substitutes. Existing 13-channel arrays are returned unchanged.
    """
    observations = np.asarray(observations)
    if observations.shape[-1] == OBSERVATION_DIM:
        return observations
    if observations.shape[-1] != 22:
        raise ValueError('Expected 13 configuration inputs or 22 archived inputs')
    return observations[..., np.r_[0:4, 5:14]]

class LearningPush(gym.Env):
    def __init__(self, reward='dense', record=False, near_fraction=0., horizon=8,
                 box_com_x=0., orientation_features='none',mass_model='halves',timestep=.005,physics=None):
        if orientation_features not in ('none','heavy_axis'):
            raise ValueError('orientation_features must be none or heavy_axis')
        self.orientation_features=orientation_features
        if not 0. <= near_fraction <= 1.:
            raise ValueError('near_fraction must lie in [0,1]')
        self.near_fraction = near_fraction
        if int(horizon)!=horizon or horizon<1:
            raise ValueError('horizon must be a positive integer')
        self.task = MultiPush(reward_kind=reward, record=record,horizon=int(horizon),box_com_x=box_com_x,mass_model=mass_model,timestep=timestep,physics=physics)
        self.action_space = spaces.Discrete(12)
        obs = self.observation()
        self.observation_space = spaces.Box(-np.inf,np.inf,obs.shape,np.float32)

    def observation(self):
        t = self.task; o = t.observation(); x,y,yaw = o['box_pose']
        c,s = np.cos(yaw),np.sin(yaw); rotation = np.array([[c,-s],[s,c]])
        error = (o['goal']-[x,y]) @ rotation
        finger = (o['pusher']-[x,y]) @ rotation
        # 2 goal-relative + 2 finger-relative + 9 configuration coordinates:
        #   0:2   target minus block x, y in block-yaw coordinates, / 0.30 m
        #   2:4   finger minus block x, y in block-yaw coordinates, / 0.50 m
        #   4:6   finger world x, y (m)
        #   6:10  block orientation quaternion w, x, y, z (world)
        #   10:13 block world x, y, z (m)
        # The executor checks settling; velocities and command count are absent.
        observation=np.r_[error/.3, finger/.5, o['state'][:9]].astype(np.float32)
        if self.orientation_features=='heavy_axis':
            # A body-fixed visible mark identifies the heavy end. Keep the full
            # 2*pi orientation, never modulo the square's pi/2 symmetry. COM
            # magnitude/inertia are NOT supplied; those remain unknown dynamics.
            sign=-1. if t.box_com_x<0 else 1.
            observation=np.r_[observation,sign*c,sign*s].astype(np.float32)
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options is None:
            if self.near_fraction > 0 and self.np_random.random() < self.near_fraction:
                angle = self.np_random.uniform(-np.pi,np.pi)
                goal = self.np_random.uniform(.04,.18)*np.array([np.cos(angle),np.sin(angle)])
            else:
                goal = self.np_random.uniform(.22,.35,2)*self.np_random.choice([-1,1],2)
            yaw = self.np_random.uniform(-.5,.5)
            options = dict(goal=goal,box_pose=(0.,0.,yaw))
        self.task.reset(**options)
        return self.observation(), {}

    def command(self, action):
        axis,sign = FACES[int(action)//3]
        return self.task.face_push(axis,sign,float(LENGTHS[int(action)%3]))

    def step(self, action):
        _,reward,term,trunc,info = self.task.step(self.command(action))
        info.update(is_success=info['success'], distance=float(np.linalg.norm(
            self.task.box_pose()[:2]-self.task.goal)), macros=self.task.count)
        # A rollout cutoff is not an absorbing task state. Learners must
        # bootstrap there, but not after success or physical failure.
        return self.observation(), reward, term, trunc, info

def summary(rows):
    return dict(success=float(np.mean([r['success'] for r in rows])),
                mean_distance=float(np.mean([r['distance'] for r in rows])),
                median_distance=float(np.median([r['distance'] for r in rows])),
                mean_macros=float(np.mean([r['macros'] for r in rows])),
                mean_return=float(np.mean([r['return_'] for r in rows])),
                invalid_actions=sum(r['invalid'] for r in rows))
