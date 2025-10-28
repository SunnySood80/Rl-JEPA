import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math
import gym
from gym import spaces


class MaskingEnv(gym.Env):
    def __init__(self, fi1_shape: tuple, mask_ratio: float = 0.5, patch_size: int = 8, device='cuda'):
        super(MaskingEnv, self).__init__()

        B, D, H8, W8 = fi1_shape  # e.g., [B, D, 28, 28]

        self.B = B
        self.D = D
        self.H8 = H8
        self.W8 = W8
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.device = device

        self.n_patches_h = H8 // patch_size
        self.n_patches_w = W8 // patch_size
        self.total_patches = self.n_patches_h * self.n_patches_w
        self.num_masked = int(mask_ratio * self.total_patches)
        
        self.action_space = spaces.Discrete(self.total_patches)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_patches_h, self.n_patches_w), dtype=np.float32)

        self.current_mask = None
        self.masked_count = 0
        self.state = None
        self.masked_patches = None
        
        self.step_count = 0
        self.max_steps = self.num_masked * 3

    def reset(self):
        self.current_mask = torch.zeros(self.H8 * self.W8, dtype=torch.bool, device=self.device)
        self.state = np.zeros((self.n_patches_h, self.n_patches_w), dtype=np.float32)
        self.masked_count = 0
        self.masked_patches = set()
        self.step_count = 0
        return self.state

    def step(self, action):
        self.step_count += 1
        
        if action in self.masked_patches:
            reward = -5
            done = (self.step_count >= self.max_steps)
            return self.state, reward, done, {}

    
        ph = action // self.n_patches_w
        pw = action % self.n_patches_w
        
        h_start = ph * self.patch_size
        h_end = min(h_start + self.patch_size, self.H8)
        w_start = pw * self.patch_size  
        w_end = min(w_start + self.patch_size, self.W8)
        
        for h in range(h_start, h_end):
            for w in range(w_start, w_end):
                self.current_mask[h * self.W8 + w] = True

        self.state[ph, pw] = 1
        self.masked_count += 1
        self.masked_patches.add(action)
        done = (self.masked_count >= self.num_masked) or (self.step_count >= self.max_steps)

        if done and self.masked_count >= self.num_masked:
            steps_saved = self.max_steps - self.step_count
            efficiency_bonus = (steps_saved / self.max_steps) * 20.0
            reward = efficiency_bonus
        else:
            reward = 0.0
    
        return self.state, reward, done, {}

    def get_final_mask(self):
        return self.current_mask


    def get_available_actions(self):

        available = []
        for action in range(self.total_patches):
            if action not in self.masked_patches:
                available.append(action)
        return available

    # def actions_to_mask(self, actions):
    #     """
    #     Convert action array to mask grid
        
    #     Args:
    #         actions: numpy array [num_patches] with 0s and 1s
        
    #     Returns:
    #         mask: [grid_h, grid_w] boolean mask
    #     """
        
    #     # Reshape flat actions to 2D grid
    #     mask = actions.reshape(self.n_patches_h, self.n_patches_w)
        
    #     # Convert to boolean if needed
    #     mask = mask.astype(bool)
        
    #     return mask