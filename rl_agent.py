import torch
import numpy as np
from PPO import PPO
from A2C import A2C
from rl_environment import MaskingEnv
from torch.distributions import Categorical


def create_custom_ppo_agent(fi1_shape, mask_ratio=0.5, patch_size=8, device='cuda'):
    env = MaskingEnv(fi1_shape, mask_ratio, patch_size, device)
    
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    agent = PPO(
        obs_shape=obs_shape,
        action_dim=action_dim,
        device=device
    )
    
    return agent, env


def create_custom_a2c_agent(fi1_shape, mask_ratio=0.5, patch_size=8, device='cuda'):
    env = MaskingEnv(fi1_shape, mask_ratio, patch_size, device)
    
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    agent = A2C(
        obs_shape=obs_shape,
        action_dim=action_dim,
        device=device
    )
    
    return agent, env


def collect_episodes_batch(agent, env, fi1_shape, batch_size=32):
    episodes = []

    for i in range(batch_size):
        obs = env.reset()
        done = False
        actions = []
        states = []
        log_probs = []
        values = []
        rewards = []
        dones = []

        while not done:
            action, log_prob, value = agent.select_action(obs, deterministic=False)
            states.append(obs.copy())
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            dones.append(done)

        final_mask = env.get_final_mask()

        episodes.append({
            'mask': final_mask,
            'actions': actions,
            'states': states,
            'log_probs': log_probs,
            'values': values,
            'rewards': rewards,
            'dones': dones
        })

    return episodes


# def collect_episodes_batch(agent, env, fi1_shape, batch_size=32):
#     episodes = []
    
#     # Get initial states for all images at once
#     states_batch = []
#     for i in range(batch_size):
#         obs = env.reset()
#         states_batch.append(obs)
    
#     states_batch = np.array(states_batch)  # [batch_size, H, W] or whatever shape
    
#     # SINGLE FORWARD PASS - get ALL decisions at once
#     states_tensor = torch.FloatTensor(states_batch).to(agent.device)
    
#     with torch.no_grad():
#         action_probs, values = agent.policy(states_tensor)  # [batch, num_patches, 2]
    
#     # Sample actions for each image
#     for i in range(batch_size):
#         # Sample ALL patch decisions for this image at once
#         dist = Categorical(action_probs[i])
        
#         # This samples decisions for ALL patches in one call
#         # If you have 512 patches, this gives you 512 decisions at once
#         actions = dist.sample()  # [num_patches]
#         log_probs = dist.log_prob(actions)  # [num_patches]
#         value = values[i].item()
        
#         # Convert actions to final mask
#         final_mask = env.actions_to_mask(actions.cpu().numpy())
        
#         # Create "episode" (but it's just 1 step now)
#         episodes.append({
#             'mask': final_mask,
#             'actions': [actions.cpu().numpy()],  # List with 1 element
#             'states': [states_batch[i]],         # List with 1 element
#             'log_probs': [log_probs.cpu().numpy()],  # List with 1 element
#             'values': [value],                   # List with 1 element
#             'rewards': [],  # Fill in later after JEPA runs
#             'dones': [True]  # Always done after 1 step
#         })
    
#     return episodes



def calculate_jepa_rewards(episodes, jepa_outputs):
    all_rewards = []
    
    for i, episode in enumerate(episodes):
        episode_rewards = []
        pixel_errors = jepa_outputs['pixel_errors'][i]
        feature_maps = jepa_outputs['features'][i]
        mask_indices = jepa_outputs['mask_indices'][i]
        
        # Map mask indices to patch positions
        valid_indices = mask_indices[mask_indices >= 0].cpu().numpy()
        
        for j, action in enumerate(episode['actions']):
            if j < len(pixel_errors) and j < len(valid_indices):
                # Pixel reward (max error)
                pixel_reward = float(pixel_errors[j]) if len(pixel_errors) > 0 else 0.0
                
                # Semantic coherence
                semantic_reward = calculate_semantic_coherence(feature_maps, action)
                
                # Combined reward
                alpha, beta = get_current_weights()
                final_reward = alpha * pixel_reward + beta * semantic_reward
            else:
                final_reward = 0.0
            
            episode_rewards.append(final_reward)
        
        all_rewards.append(episode_rewards)
    
    return all_rewards


def calculate_semantic_coherence(feature_maps, action):
    try:
        if torch.is_tensor(feature_maps):
            feature_maps = feature_maps.detach().cpu().float()
        
        n_patches_h, n_patches_w = feature_maps.shape[-2:]
        ph = action // n_patches_w
        pw = action % n_patches_w
        
        if ph >= n_patches_h or pw >= n_patches_w:
            return 0.0
        
        masked_patch_features = feature_maps[ph, pw]
        
        h_start = max(0, ph - 1)
        h_end = min(n_patches_h, ph + 2)
        w_start = max(0, pw - 1)
        w_end = min(n_patches_w, pw + 2)
        
        neighborhood_features = []
        for h in range(h_start, h_end):
            for w in range(w_start, w_end):
                if h != ph or w != pw:
                    neighborhood_features.append(feature_maps[h, w])
        
        if len(neighborhood_features) == 0:
            return 0.0
        
        avg_neighbor_features = torch.stack(neighborhood_features).mean(dim=0)
        semantic_score = torch.cosine_similarity(masked_patch_features, avg_neighbor_features, dim=0)
        
        return float(semantic_score.item())
    
    except Exception as e:
        print(f"Semantic coherence calculation failed: {e}")
        return 0.0


def get_current_weights(current_step=None, total_steps=None, 
                       pixel_weight_multiplier=1.0,
                       semantic_weight_multiplier=1.0):
    
    alpha = 0.5 * pixel_weight_multiplier
    beta = 0.5 * semantic_weight_multiplier
    
    total = alpha + beta
    if total > 0:
        alpha, beta = alpha/total, beta/total
    else:
        alpha, beta = 0.5, 0.5
    
    return alpha, beta

def rl_generate_mask(agent, fi1_shape, mask_ratio=0.5, patch_size=8, device='cuda'):
    env = MaskingEnv(fi1_shape, mask_ratio, patch_size, device)

    obs = env.reset()
    done = False
    
    while not done:
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

    return env.get_final_mask()

class MaskingAgentTrainer:

    def __init__(self, fi1_shape, mask_ratio=0.5, patch_size=8, device='cuda'):
        
        self.agent, self.env = create_custom_a2c_agent(fi1_shape, mask_ratio, patch_size, device)
        self.fi1_shape = fi1_shape

    def generate_masks_for_batch(self, batch_size=32):


        episodes = collect_episodes_batch(self.agent, self.env, self.fi1_shape, batch_size)

        masks = [ep['mask'] for ep in episodes]

        return masks, episodes
    
    def update_agent(self, episodes, jepa_outputs):
        jepa_rewards = calculate_jepa_rewards(episodes, jepa_outputs)
        
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_values = []
        all_rewards = []
        all_dones = []
        
        for i, episode in enumerate(episodes):
            episode_rewards = jepa_rewards[i]
            
            all_states.extend(episode['states'])
            all_actions.extend(episode['actions'])
            all_old_log_probs.extend(episode['log_probs'])
            all_values.extend(episode['values'])
            all_rewards.extend(episode_rewards)
            all_dones.extend(episode['dones'])
        
        advantages, returns = self.agent.compute_gae(
            rewards=all_rewards,
            values=all_values,
            dones=all_dones
        )
        
        self.agent.train_on_batch(
            states=all_states,
            actions=all_actions,
            old_log_probs=all_old_log_probs,
            advantages=advantages,
            returns=returns
        )
