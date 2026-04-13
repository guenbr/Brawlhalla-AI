import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from torch.distributions import Categorical
from run.eight_val.environment import BrawlhallaEnv, ACTION_NAMES, NUM_ACTIONS, OBS_SIZE, \
    STARTING_LIVES

# hyperparameters
LEARNING_RATE = 0.00015
GAMMA = 0.995
LAMBDA = 0.95
CLIP_EPSILON = 0.10
ENTROPY_COEF = 0.01
DIVERSITY_COEF = 0.03
VALUE_COEF = 1.0
UPDATE_EPOCHS = 4
EPISODES_PER_UPDATE = 8

# softly pushes policy toward a balanced playstyle during updates
TARGET_ACTION_DIST = torch.tensor([
    0.08,  # neutral
    0.10,  # move_left
    0.10,  # move_right
    0.10,  # jump
    0.12,  # light
    0.10,  # heavy
    0.08,  # dodge
    0.14,  # left_heavy
    0.14,  # right_heavy
    0.07,  # left_light
    0.07,  # right_light
], dtype=torch.float32)

CHECKPOINT_DIR = "run/eight_val/checkpoints"
LOG_DIR = "run/eight_val/logs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic network that outputs action probabilities and a state value.

    Args:
        obs_size: Length of the flattened observation vector.
        num_actions: Number of discrete actions.
    """

    def __init__(self, obs_size: int, num_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        """Runs a forward pass and returns action probabilities and value estimate.

        Args:
            x: Float tensor of shape (batch, obs_size).

        Returns:
            Tuple of (action_probs, value) tensors.
        """
        feat = self.shared(x)
        return self.actor(feat), self.critic(feat)


class PPOMemory:
    """Stores experience tuples collected during rollouts for PPO updates."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def store(self, state, action, reward, value, log_prob, done):
        """Appends a single transition to memory.

        Args:
            state: Observation array.
            action: Integer action taken.
            reward: Scalar reward received.
            value: Critic value estimate for this state.
            log_prob: Log probability of the action taken.
            done: Whether this step ended the episode.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        """Clears all stored experience."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def get_batches(self):
        """Returns all stored experience as numpy arrays.

        Returns:
            Tuple of (states, actions, rewards, values, log_probs, dones) arrays.
        """
        return (np.array(self.states),
                np.array(self.actions),
                np.array(self.rewards),
                np.array(self.values),
                np.array(self.log_probs),
                np.array(self.dones))


def compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    """Computes generalized advantage estimates and discounted returns.

    Args:
        rewards: List of scalar rewards.
        values: List of critic value estimates.
        dones: List of done flags.
        gamma: Discount factor.
        lam: GAE lambda for bias-variance tradeoff.

    Returns:
        Tuple of (advantages, returns) lists.
    """
    advantages, gae = [], 0
    for t in reversed(range(len(rewards))):
        nxt = 0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def run_ppo_update(model, optimizer, memory, device):
    """Runs PPO update epochs on the collected memory batch.

    Args:
        model: ActorCritic instance.
        optimizer: Adam optimizer.
        memory: PPOMemory instance with stored experience.
        device: torch device.

    Returns:
        Tuple of (loss, entropy) scalar values from the last epoch.
    """
    states_b, actions_b, rewards_b, values_b, old_lp_b, dones_b = memory.get_batches()

    advantages, returns = compute_gae(rewards_b, values_b, dones_b)
    advantages = np.array(advantages)
    returns = np.array(returns)

    # normalize advantages for stable training
    if advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = advantages - advantages.mean()

    target_dist = TARGET_ACTION_DIST.to(device)
    loss_val = 0.0
    ent_val = 0.0

    for _ in range(UPDATE_EPOCHS):
        s_t = torch.FloatTensor(states_b).to(device)
        a_t = torch.LongTensor(actions_b).to(device)
        olp_t = torch.FloatTensor(old_lp_b).to(device)
        adv_t = torch.FloatTensor(advantages).to(device)
        ret_t = torch.FloatTensor(returns).to(device)

        probs, vals = model(s_t)
        probs = torch.clamp(probs, min=1e-6, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        dist = Categorical(probs)
        nlp = dist.log_prob(a_t)
        ent = dist.entropy().mean()

        ratio = torch.exp(nlp - olp_t)
        s1 = ratio * adv_t
        s2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * adv_t

        actor_loss = -torch.min(s1, s2).mean()
        critic_loss = nn.MSELoss()(vals.squeeze(), ret_t)

        # KL penalty to keep action distribution close to target
        mean_probs = probs.mean(dim=0)
        mean_probs = mean_probs / mean_probs.sum()
        kl_to_target = (target_dist * (target_dist.log() - mean_probs.log())).sum()

        loss = (actor_loss
                + VALUE_COEF * critic_loss
                - ENTROPY_COEF * ent
                + DIVERSITY_COEF * kl_to_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loss_val = loss.item()
        ent_val = ent.item()

    return loss_val, ent_val


def train():
    """Main training loop — runs PPO episodes indefinitely until interrupted."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    print("BRAWLHALLA PPO — 8 VALUE STATE")
    print("=" * 70)

    env = BrawlhallaEnv()
    model = ActorCritic(obs_size=OBS_SIZE, num_actions=NUM_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    memory = PPOMemory()

    p1_wins = 0
    p2_wins = 0
    episode_offset = 0

    # resume from latest checkpoint if one exists
    latest_path = os.path.join(CHECKPOINT_DIR, "ppo_latest.pth")
    if os.path.exists(latest_path):
        print(f"\nLoading checkpoint: {latest_path}")
        ckpt = torch.load(latest_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for group in optimizer.param_groups:
            group['lr'] = LEARNING_RATE
        p1_wins = ckpt.get('p1_wins', 0)
        p2_wins = ckpt.get('p2_wins', 0)
        episode_offset = ckpt.get('episode', 0)
        print(f"Resumed from episode {episode_offset} | "
              f"P1 wins: {p1_wins} | CPU wins: {p2_wins}\n")
    else:
        print("No checkpoint found — starting fresh.\n")

    episode_batch = 0
    loss_val = 0.0
    ent_val = 0.0
    episode = 0

    try:
        while True:
            global_ep = episode_offset + episode + 1

            combined_data = env.reset()
            obs = env.get_obs(combined_data)
            done = False

            ep_reward = 0.0
            ep_steps = 0
            deaths_this_ep = 0
            action_counts = [0] * NUM_ACTIONS
            final_probs = None

            print(f"\n{'=' * 70}")
            print(f"EPISODE {global_ep}  |  entropy_coef={ENTROPY_COEF}  |  lives={STARTING_LIVES}")
            print(f"{'=' * 70}")

            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

                with torch.no_grad():
                    action_probs, value = model(obs_t)
                    action_probs = torch.clamp(action_probs, min=1e-6, max=1.0)
                    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

                    dist_cat = Categorical(action_probs)
                    action = dist_cat.sample()
                    log_prob = dist_cat.log_prob(action)

                action_idx = action.item()
                print(f"  step {ep_steps} | {ACTION_NAMES[action_idx]}")
                action_counts[action_idx] += 1
                final_probs = action_probs.cpu().numpy()[0]

                next_obs, next_cd, reward, done, info = env.step(action_idx)
                memory.store(obs, action_idx, reward, value.item(), log_prob.item(), done)

                obs = next_obs
                combined_data = next_cd
                ep_reward += reward
                ep_steps += 1

                if info['is_player_dead']:
                    deaths_this_ep += 1

                if ep_steps % 50 == 0:
                    probs_np = action_probs.cpu().numpy()[0]
                    ent_now = -np.sum(probs_np * np.log(probs_np + 1e-8))
                    max_prob = probs_np.max()
                    print(f"  [step {ep_steps}] action={ACTION_NAMES[action_idx]:<12} | "
                          f"probs: {np.round(probs_np, 3)} | "
                          f"V={value.item():.2f} | ent={ent_now:.3f} | max_p={max_prob:.3f}")
                    if ent_now < 0.5:
                        print(f"  !! ENTROPY WARNING: {ent_now:.4f} — policy collapsing")
                    if max_prob > 0.60:
                        print(f"  !! SPAM WARNING: one action at {max_prob:.1%}")

                if ep_steps > 30000:
                    print("\nEpisode timeout")
                    done = True

            # episode end — update win counters and print summary
            env.controls.release_all()
            if info['lives'][0] > info['lives'][1]:
                p1_wins += 1
            elif info['lives'][1] > info['lives'][0]:
                p2_wins += 1

            print(f"\n  Action counts:")
            for name, count in zip(ACTION_NAMES, action_counts):
                pct = 100 * count / max(ep_steps, 1)
                flag = " <-- SPAM" if pct > 40 else ""
                print(f"    {name:<12} {count:>5}  ({pct:.1f}%){flag}")

            print(f"\n{'=' * 70}")
            print(f"EPISODE {global_ep} COMPLETE")
            print(f"Reward: {ep_reward:.2f} | Steps: {ep_steps} | Deaths: {deaths_this_ep}")
            print(f"Final: P1={int(info['lives'][0])} lives, CPU={int(info['lives'][1])} lives")
            print(f"Win Rate: P1={p1_wins}/{global_ep} ({100 * p1_wins / global_ep:.1f}%)")

            episode += 1
            episode_batch += 1

            # run PPO update every N episodes
            if episode_batch >= EPISODES_PER_UPDATE:
                if len(memory.states) > 0:
                    print(f"\nTraining on {len(memory.states)} experiences "
                          f"({episode_batch} episodes)...")
                    loss_val, ent_val = run_ppo_update(model, optimizer, memory, device)
                    print(f"Loss: {loss_val:.4f} | Entropy: {ent_val:.4f}")

                    if ent_val < 0.5:
                        print("!! Entropy collapsed — check reward scale / lr")
                    elif ent_val > 1.85:
                        print("!! Entropy still near-random — policy not converging yet")

                    memory.clear()
                episode_batch = 0

                # save numbered checkpoint and overwrite latest
                ckpt_data = {
                    'episode': global_ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'p1_wins': p1_wins,
                    'p2_wins': p2_wins,
                }
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"ppo_ep{global_ep}.pth")
                torch.save(ckpt_data, ckpt_path)
                torch.save(ckpt_data, os.path.join(CHECKPOINT_DIR, "ppo_latest.pth"))
                print(f"Checkpoint saved: {ckpt_path}")

            # log episode stats to CSV
            csv_path = os.path.join(LOG_DIR, "training_metrics.csv")
            write_hdr = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
            with open(csv_path, 'a') as f:
                if write_hdr:
                    count_hdr = ','.join(f'count_{n}' for n in ACTION_NAMES)
                    prob_hdr = ','.join(f'prob_{n}' for n in ACTION_NAMES)
                    f.write(f'timestamp,episode,deaths,reward,steps,'
                            f'p1_lives,cpu_lives,p1_wins,p2_wins,entropy,'
                            f'{count_hdr},{prob_hdr}\n')

                counts_str = ','.join(str(c) for c in action_counts)
                probs_str = ','.join(f'{p:.4f}' for p in final_probs) \
                    if final_probs is not None else ','.join(['0.0000'] * NUM_ACTIONS)
                f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")},'
                        f'{global_ep},{deaths_this_ep},{ep_reward:.2f},'
                        f'{ep_steps},{int(info["lives"][0])},'
                        f'{int(info["lives"][1])},{p1_wins},{p2_wins},'
                        f'{ent_val:.4f},{counts_str},{probs_str}\n')

    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
        torch.save({
            'episode': global_ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'p1_wins': p1_wins,
            'p2_wins': p2_wins,
        }, os.path.join(CHECKPOINT_DIR, "ppo_latest.pth"))
        print("Final checkpoint saved.")


if __name__ == "__main__":
    train()
