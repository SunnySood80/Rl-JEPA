import torch
import numpy as np
from PPO import PPO, ExperienceBuffer
from rl_environment import MaskingEnv


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
            available = env.get_available_actions()
            action, log_prob, value = agent.select_action(obs, available, deterministic=False)
            states.append(obs.copy())
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            
            # Debug: Check if policy is learning (first episode, first action only)
            if i == 0 and len(actions) == 1:
                state_tensor = torch.FloatTensor(obs).to(agent.device)
                action_probs, _ = agent.policy.forward(state_tensor)
                max_prob = action_probs.max().item()
                uniform_prob = 1.0 / action_probs.shape[-1]
                entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8)).item()
                is_uniform = abs(max_prob - uniform_prob) < 0.05 * uniform_prob
                print(f"[RL POLICY] max_prob={max_prob:.4f} (uniform={uniform_prob:.4f}), entropy={entropy:.4f}, still_uniform={is_uniform}")
            
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


def calculate_jepa_rewards(episodes, jepa_outputs):
    """
    Calculate rewards for RL agent based on JEPA reconstruction errors.
    
    Args:
        episodes: List of episode dicts with 'actions', 'states', etc.
        jepa_outputs: Dict with 'pixel_errors', 'features', 'mask_indices'
    
    Returns:
        all_rewards: List of reward lists, one per episode
    """
    all_rewards = []
    
    for i, episode in enumerate(episodes):
        episode_rewards = []
        
        # Get JEPA outputs for this episode (already on CPU from train.py batch transfer)
        pixel_errors = jepa_outputs['pixel_errors'][i]  # [N] reconstruction errors per pixel (numpy)
        feature_maps_cpu = jepa_outputs['features'][i]   # [D, H8, W8] already on CPU
        mask_indices = jepa_outputs['mask_indices'][i]  # [M] already on CPU
        
        # Get valid mask indices (filter out padding -1s) - already on CPU
        valid_indices = mask_indices[mask_indices >= 0].numpy()
        
        # Create mapping from pixel position → reconstruction error
        pixel_pos_to_error = {}
        for idx, pixel_pos in enumerate(valid_indices):
            if idx < len(pixel_errors):
                pixel_pos_to_error[int(pixel_pos)] = float(pixel_errors[idx])
        
        # ✅ GET GRID DIMENSIONS FROM FEATURE_MAPS
        # feature_maps shape is [D, H8, W8] where H8, W8 are pixel dimensions at stride 8
        patch_size = 8  # From your environment
        D, H8, W8 = feature_maps_cpu.shape
        n_patches_h = H8 // patch_size  # Number of patches vertically
        n_patches_w = W8 // patch_size  # Number of patches horizontally
        
        # CRITICAL OPTIMIZATION: Reshape feature_maps ONCE per episode, not per action
        # Reshape from [D, H8, W8] to [n_patches_h, n_patches_w, D] by averaging over patch pixels
        feature_maps_reshaped = feature_maps_cpu.view(D, n_patches_h, patch_size, n_patches_w, patch_size)
        feature_maps_patches = feature_maps_reshaped.mean(dim=(2, 4))  # [D, n_patches_h, n_patches_w]
        feature_maps_patches = feature_maps_patches.permute(1, 2, 0)  # [n_patches_h, n_patches_w, D]
        
        # For each action, find its pixels and aggregate their errors
        for j, action in enumerate(episode['actions']):
            # Convert action (patch ID) to patch coordinates
            ph = action // n_patches_w
            pw = action % n_patches_w
            
            # Check if action is valid
            if ph >= n_patches_h or pw >= n_patches_w:
                print(f"[ERROR] Invalid action {action}: patch ({ph},{pw}) out of bounds!")
                final_reward = 0.0
                episode_rewards.append(final_reward)
                continue
            
            # Get pixel range for this patch
            h_start = ph * patch_size
            h_end = min(h_start + patch_size, H8)
            w_start = pw * patch_size
            w_end = min(w_start + patch_size, W8)
            
            # Collect errors for all pixels in this patch
            patch_errors = []
            for h in range(h_start, h_end):
                for w in range(w_start, w_end):
                    pixel_pos = h * W8 + w
                    if pixel_pos in pixel_pos_to_error:
                        patch_errors.append(pixel_pos_to_error[pixel_pos])
            
            # Aggregate errors (mean)
            if len(patch_errors) > 0:
                pixel_reward = float(np.mean(patch_errors))
            else:
                # This patch wasn't actually masked (duplicate action or error)
                if j < 5 or i == 0:  # Debug first few
                    print(f"[WARNING] Action {action} (patch {ph},{pw}) has no matching pixel errors!")
                pixel_reward = 0.0
            
            # DEBUG first 3 actions of first episode
            # if i == 0 and j < 3:
            #     print(f"  Action[{j}]={action} (patch {ph},{pw}):")
            #     print(f"    Pixels: [{h_start}:{h_end}, {w_start}:{w_end}]")
            #     print(f"    Pixel positions: {h_start*W8+w_start} to {(h_end-1)*W8+(w_end-1)}")
            #     print(f"    Errors found: {len(patch_errors)}/{(h_end-h_start)*(w_end-w_start)} pixels")
            #     print(f"    Mean error: {pixel_reward:.4f}")
            
            # Semantic coherence (pass already-reshaped feature_maps_patches)
            semantic_reward = calculate_semantic_coherence(feature_maps_patches, action, n_patches_h, n_patches_w)
            
            # Combined reward
            alpha, beta = get_current_weights()
            final_reward = alpha * pixel_reward + beta * semantic_reward
            # if i == 0 and j < 1:
            #     print(f"[RL] Example reward breakdown: Pixel {pixel_reward:.4f} | Semantic {semantic_reward:.4f} | Final {final_reward:.4f}")
            
            episode_rewards.append(final_reward)
        
        all_rewards.append(episode_rewards)
    
    return all_rewards


def calculate_semantic_coherence(feature_maps_patches, action, n_patches_h, n_patches_w, from_radius=2):
    """
    Calculate semantic coherence reward for a given action.
    
    Args:
        feature_maps_patches: [n_patches_h, n_patches_w, D] already reshaped feature maps
        action: Patch ID
        n_patches_h: Number of patches vertically
        n_patches_w: Number of patches horizontally
    """
    try:
        ph = action // n_patches_w
        pw = action % n_patches_w
        
        if ph >= n_patches_h or pw >= n_patches_w:
            return 0.0
        
        masked_patch_features = feature_maps_patches[ph, pw]

        # from_radius is the left right, up and down patches from the selected patch        
        
        h_start = max(0, ph - from_radius)
        h_end = min(n_patches_h, ph + from_radius + 1)
        w_start = max(0, pw - from_radius)
        w_end = min(n_patches_w, pw + from_radius + 1)
        
        neighborhood_features = []
        for h in range(h_start, h_end):
            for w in range(w_start, w_end):
                if h != ph or w != pw:
                    neighborhood_features.append(feature_maps_patches[h, w])
        
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
    
    # Favor pixel rewards more (they're more discriminative)
    # Pixel: 0.7, Semantic: 0.3 (was 0.5/0.5)
    alpha = 0.7 * pixel_weight_multiplier
    beta = 0.3 * semantic_weight_multiplier
    
    total = alpha + beta
    if total > 0:
        alpha, beta = alpha/total, beta/total
    else:
        alpha, beta = 0.7, 0.3
    
    return alpha, beta


class MaskingAgentTrainer:

    def __init__(self, fi1_shape, mask_ratio=0.5, patch_size=8, device='cuda'):
        
        self.agent, self.env = create_custom_ppo_agent(fi1_shape, mask_ratio, patch_size, device)
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
        
        # Debug: Check reward variance
        all_rewards_flat = []
        for i, episode in enumerate(episodes):
            episode_rewards = jepa_rewards[i]
            all_rewards_flat.extend(episode_rewards)
        
        if len(all_rewards_flat) > 0:
            reward_mean = np.mean(all_rewards_flat)
            reward_std = np.std(all_rewards_flat)
            reward_min = np.min(all_rewards_flat)
            reward_max = np.max(all_rewards_flat)
            # if len(episodes) > 0 and len(episodes[0]['actions']) > 0:
            #     print(f"[RL DEBUG] Reward stats: mean={reward_mean:.4f}, std={reward_std:.4f}, min={reward_min:.4f}, max={reward_max:.4f}")
        
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
