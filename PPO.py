
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict


class PolicyNetwork(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int], action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_h, self.obs_w = obs_shape
        self.input_dim = self.obs_h * self.obs_w
        
        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        if state.dim() == 2:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = state.shape[0]
        state_flat = state.reshape(batch_size, -1)
        
        features = self.shared(state_flat)
        
        action_logits = self.policy_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        value = self.value_head(features)
        
        if squeeze_output:
            action_probs = action_probs.squeeze(0)
            value = value.squeeze(0)
            
        return action_probs, value
    
    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            action_probs, value = self.forward(state)
            
            if deterministic:
                action = torch.argmax(action_probs)
            else:
                dist = Categorical(action_probs)
                action = dist.sample()
                
        return action.item(), action_probs, value.item()


class PPO:
    def __init__(
        self,
        obs_shape: Tuple[int, int],
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 2,
        batch_size: int = 64,
        device: str = 'cuda'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.policy = PolicyNetwork(obs_shape, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        state_tensor = torch.FloatTensor(state).to(self.device)
        action, action_probs, value = self.policy.get_action(state_tensor, deterministic)
        
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
        
        return action, log_prob, value
    
    def compute_gae(
        self, 
        rewards: List[float], 
        values: List[float], 
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages = []
        gae = 0
        
        values = values + [0]
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                
            advantages.insert(0, gae)
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(values[:-1], dtype=np.float32)
        
        return advantages, returns
    
    def train_on_batch(
        self,
        states: List[np.ndarray],
        actions: List[int],
        old_log_probs: List[float],
        advantages: np.ndarray,
        returns: np.ndarray
    ):
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        dataset_size = len(states)
        for epoch in range(self.n_epochs):
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                action_probs, values = self.policy(batch_states)
                
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                values_flat = values.squeeze(-1)
                if values_flat.shape != batch_returns.shape:
                    batch_returns = batch_returns.view_as(values_flat)
                value_loss = F.mse_loss(values_flat, batch_returns)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class ExperienceBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def add(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def get(self):
        return {
            'states': self.states,
            'actions': self.actions,
            'log_probs': self.log_probs,
            'values': self.values,
            'rewards': self.rewards,
            'dones': self.dones
        }
    
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def __len__(self):
        return len(self.states)