import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from src.screen_grab.grab import ScreenGrab
from src.health_api.health import HealthAPI
from src.controls.controls import Controls
import time
import os
from src.player_location.player_detector import PlayerDetector

STARTING_LIVES = 15

ACTION_NAMES = [
    'neutral',
    'move_left',
    'move_right',
    'jump',
    'light',
    'heavy',
    'dodge',
    'left_heavy',
    'right_heavy',
    'left_light',
    'right_light',
]

NUM_ACTIONS = len(ACTION_NAMES)

TARGET_ACTION_DIST = torch.tensor([
    0.08,   # neutral
    0.10,   # move_left
    0.10,   # move_right
    0.10,   # jump
    0.12,   # light
    0.10,   # heavy
    0.08,   # dodge
    0.14,   # left_heavy
    0.14,   # right_heavy
    0.07,   # left_light
    0.07,   # right_light
], dtype=torch.float32)

# flat layout: [h0, l0, x0, y0, h1, l1, x1, y1, dx, dy, dist, dist_left, dist_right, on_platform]
# 8 (scraped+location) + 6 (derived) = 14
COMBINED_DATA_SIZE = 14


class ActorCritic(nn.Module):
    def __init__(self, input_size=COMBINED_DATA_SIZE, num_actions=NUM_ACTIONS):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, combined_data):
        # Input is already flat (batch, 14) — no reshape needed
        x = combined_data
        return self.actor(x), self.critic(x)


class PPOMemory:
    def __init__(self):
        self.combined_data = []
        self.actions       = []
        self.rewards       = []
        self.values        = []
        self.log_probs     = []
        self.dones         = []

    def store(self, combined_data, action, reward, value, log_prob, done):
        self.combined_data.append(combined_data)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.combined_data = []; self.actions = []
        self.rewards = []; self.values = []; self.log_probs = []; self.dones = []

    def get_batches(self):
        return (np.array(self.combined_data),
                np.array(self.actions),
                np.array(self.rewards),
                np.array(self.values),
                np.array(self.log_probs),
                np.array(self.dones))


class BrawlhallaEnv:
    def __init__(self, monitor=0, frame_skip=2, starting_lives=STARTING_LIVES):
        self.screen             = ScreenGrab(monitor=monitor)
        self.health_api         = HealthAPI(starting_lives=starting_lives)
        self.player_detector    = PlayerDetector(monitor=monitor)
        self.controls           = Controls()
        self.starting_lives     = starting_lives
        self.prev_health        = np.array([100.0, 100.0])
        self.frame_skip         = frame_skip
        self.first_reset        = True
        self.recent_actions     = deque(maxlen=20)
        self.episode_start_time = None
        self.prev_combined_data = np.zeros(14, dtype=np.float32)  # flat shape

    def reset(self):
        if self.first_reset:
            self.first_reset = False
        else:
            print("\nResetting game")
            self.controls.release_all()
            self.controls.reset_game()
            self.health_api.health               = np.array([100.0, 100.0])
            self.health_api.lives                = np.array([self.starting_lives,
                                                              self.starting_lives])
            self.health_api.last_valid_health_p1 = 100
            self.health_api.last_valid_health_p2 = 100
            print("Game reset complete")

        self.prev_health        = np.array([100.0, 100.0])
        self.recent_actions.clear()
        self.episode_start_time = time.time()
        combined_data, _, _ = self.capture_frame()
        self.prev_combined_data = combined_data.copy()

        self.reward_components = {
            'damage_dealt': 0.0,
            'damage_taken': 0.0,
            'time_penalty': 0.0,
            'offstage_penalty': 0.0,
            'suicide_penalty': 0.0,
            'death_penalty': 0.0,
            'kill_reward': 0.0,
        }

        return combined_data

    def capture_frame(self):
        full_frame = self.screen.grab(greyscale=False)

        health_vector, is_player_dead, winner, lives, is_game_over = \
            self.health_api.process_frame(full_frame)
        location_matrix = self.player_detector.get_positions(full_frame)

        normalized_health = health_vector / 100.0
        normalized_lives = lives / float(self.starting_lives)

        location_matrix = np.array(location_matrix, dtype=np.float32)
        location_matrix[:, 0] = location_matrix[:, 0] / 2560.0
        location_matrix[:, 1] = location_matrix[:, 1] / 1440.0
        location_matrix = np.clip(location_matrix, 0, 1)

        if (lives[0] <= 0 or lives[1] <= 0) and not is_game_over:
            print("Forcing game over (a player's lives reached 0)")
            is_game_over = True

        PLATFORM_LEFT  = 0.319
        PLATFORM_RIGHT = 0.678
        PLATFORM_Y     = 0.581

        p1_x, p1_y = location_matrix[0, 0], location_matrix[0, 1]
        p2_x, p2_y = location_matrix[1, 0], location_matrix[1, 1]

        dx = p2_x - p1_x
        dy = p2_y - p1_y
        dist_to_opponent = np.sqrt(dx ** 2 + dy ** 2)
        dist_left_edge   = p1_x - PLATFORM_LEFT
        dist_right_edge  = PLATFORM_RIGHT - p1_x
        on_platform      = float((dist_left_edge > 0) and (dist_right_edge > 0) and (p1_y <= PLATFORM_Y))

        derived = np.array([
            dx, dy,
            dist_to_opponent,
            dist_left_edge,
            dist_right_edge,
            on_platform,
        ], dtype=np.float32)

        scraped_data = np.stack([normalized_health, normalized_lives], axis=0).T
        # 2D combined for reference (2, 4): [health, lives, x, y] per player
        combined_2d  = np.concatenate([scraped_data, location_matrix], axis=1)

        # Flat layout (14,): [h0,l0,x0,y0, h1,l1,x1,y1, dx,dy,dist,dist_left,dist_right,on_platform]
        combined_data = np.concatenate([combined_2d.flatten(), derived])

        return combined_data, is_player_dead, is_game_over

    def step(self, action):
        total_reward = 0

        for _ in range(self.frame_skip):
            self.controls.execute_action(action)
            time.sleep(0.0089)

        combined_data, is_player_dead, is_game_over = self.capture_frame()

        # flat indices: h0=0, l0=1, x0=2, y0=3, h1=4, l1=5, x1=6, y1=7
        health = combined_data[[0, 4]] * 100.0
        lives  = combined_data[[1, 5]] * float(self.starting_lives)

        total_reward += self.calculate_reward(health, lives, is_player_dead,
                                              is_game_over, combined_data, action)

        if is_player_dead:
            snap_p1 = int(lives[0])
            snap_p2 = int(lives[1])
            print("Death detected - monitoring respawn period for additional deaths...")

            for check_num in range(26):
                time.sleep(0.1)
                full_frame = self.screen.grab(greyscale=False)
                _, temp_dead, _, temp_lives_raw, _ = \
                    self.health_api.process_frame(full_frame)

                if self.health_api.is_game_over():
                    break

                if temp_dead:
                    cur_p1 = int(temp_lives_raw[0])
                    cur_p2 = int(temp_lives_raw[1])
                    if cur_p1 < snap_p1 or cur_p2 < snap_p2:
                        temp_health = self.health_api.health.copy()
                        add_r = self.calculate_reward(
                            temp_health, temp_lives_raw, True, False, combined_data, action)
                        total_reward += add_r
                        print(f"  Additional death at {0.1*(check_num+1):.1f}s | "
                              f"P1: {snap_p1}->{cur_p1}  P2: {snap_p2}->{cur_p2} | "
                              f"reward: {add_r:.1f}")
                        health  = temp_health
                        lives   = temp_lives_raw
                        snap_p1 = cur_p1
                        snap_p2 = cur_p2

            self.health_api.last_valid_health_p1 = 100
            self.health_api.last_valid_health_p2 = 100
            self.prev_combined_data = combined_data.copy()
            self.prev_health        = np.array([100.0, 100.0])
        else:
            self.prev_health = health.copy()

        info = {'health': health, 'lives': lives,
                'winner': None, 'is_player_dead': is_player_dead}
        return combined_data, total_reward, is_game_over, info

    def calculate_reward(self, health, lives, is_player_dead, is_game_over, combined_data, action):
        reward = 0

        health_diff  = health - self.prev_health
        damage_dealt = abs(health_diff[1]) if health_diff[1] < 0 else 0
        damage_taken = abs(health_diff[0]) if health_diff[0] < 0 else 0

        # flat indices: x0=2, y0=3
        p1_x = combined_data[2]
        p1_y = combined_data[3]
        on_platform = (0.319 < p1_x < 0.678) and (p1_y <= 0.581)

        offstage_pen = -.6 if not on_platform else 0.0
        dealt_r = damage_dealt * 0.05
        taken_r = -(damage_taken * 0.025)

        reward += offstage_pen + dealt_r + taken_r

        self.reward_components['offstage_penalty'] += offstage_pen
        self.reward_components['damage_dealt']     += dealt_r
        self.reward_components['damage_taken']     += taken_r

        if is_player_dead:
            print(f'  P1 death | damage_taken={damage_taken:.1f}')
            if damage_taken > 50:
                reward -= 10
                self.reward_components['suicide_penalty'] -= 10
                print('suicide reward')
            if health[0] <= 1:
                reward -= 3
                self.reward_components['death_penalty'] -= 3
            if health[1] <= 1:
                reward += 40
                self.reward_components['kill_reward'] += 10
                print('kill reward')

        self.prev_combined_data = combined_data.copy()
        return reward


def compute_gae(rewards, values, dones, gamma=0.995, lam=0.95):
    advantages, gae = [], 0
    for t in reversed(range(len(rewards))):
        nxt   = 0 if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt * (1 - dones[t]) - values[t]
        gae   = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def train_ppo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    os.makedirs('checkpoints_14', exist_ok=True)
    os.makedirs('logs_14', exist_ok=True)

    env       = BrawlhallaEnv(monitor=1, frame_skip=2, starting_lives=STARTING_LIVES)
    model     = ActorCritic(input_size=COMBINED_DATA_SIZE, num_actions=NUM_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00015)
    controls  = Controls()
    memory    = PPOMemory()

    p1_wins        = 0
    p2_wins        = 0
    episode_offset = 0

    checkpoint_path = 'checkpoints_14/ppo_latest.pth'
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
       # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for group in optimizer.param_groups:
            group['lr'] = 0.00015
        p1_wins        = ckpt.get('p1_wins', 0)
        p2_wins        = ckpt.get('p2_wins', 0)
        episode_offset = ckpt.get('episode', 0)
        print(f"Resumed from episode {episode_offset} | "
              f"P1 wins: {p1_wins} | P2 wins: {p2_wins}\n")
    else:
        print("Starting fresh\n")

    gamma               = 0.965
    lam                 = 0.95
    epsilon             = 0.20
    epochs_per_update   = 4
    entropy_coef        = 0.003
    diversity_coef      = 0.0
    episodes_per_update = 4
    num_episodes        = 500

    episode_batch = 0
    loss_val      = 0.0
    ent_val       = 0.0

    for episode in range(num_episodes):
        global_episode = episode_offset + episode + 1

        combined_data = env.reset()
        episode_reward = 0
        episode_steps  = 0
        deaths_this_ep = 0
        action_counts  = [0] * NUM_ACTIONS
        final_probs    = None

        print(f"\n{'='*70}")
        print(f"EPISODE {global_episode} (session ep {episode+1}/{num_episodes})  |  "
              f"entropy_coef={entropy_coef:.4f}  |  lives={STARTING_LIVES}")
        print(f"{'='*70}")

        while True:
            cd_t = torch.FloatTensor(combined_data).unsqueeze(0).to(device)

            with torch.no_grad():
                action_probs, value = model(cd_t)
                action_probs = torch.clamp(action_probs, min=1e-6, max=1.0)
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

                # --- edge action masking ---
                dist_left = combined_data[11]
                dist_right = combined_data[12]
                EDGE_THRESHOLD = 0.15

                mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                #       neutral, left, right, jump, light, heavy, dodge, l_heavy, r_heavy, l_light, r_light

                if dist_right < EDGE_THRESHOLD:
                    mask[2] = 0  # move_right
                    mask[5] = 0  # heavy
                    mask[8] = 0  # right_heavy
                    mask[10] = 0  # right_light

                if dist_left < EDGE_THRESHOLD:
                    mask[1] = 0  # move_left
                    mask[5] = 0  # heavy
                    mask[7] = 0  # left_heavyl
                    mask[9] = 0  # left_light

                mask_t = torch.FloatTensor(mask).to(device)
                action_probs = action_probs * mask_t
                action_probs = action_probs + 1e-6  # ensure no zeros after masking
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            action_counts[action.item()] += 1
            final_probs = action_probs.cpu().numpy()[0]

            next_cd, reward, done, info = env.step(action.item())
            memory.store(combined_data, action.item(), reward,
                         value.item(), log_prob.item(), done)

            combined_data  = next_cd
            episode_reward += reward
            episode_steps  += 1

            if info['is_player_dead']:
                deaths_this_ep += 1

            if episode_steps % 50 == 0:
                probs_np = action_probs.cpu().numpy()[0]
                ent_now  = -np.sum(probs_np * np.log(probs_np + 1e-8))
                max_prob = probs_np.max()
                print(f"  [step {episode_steps}] probs: {np.round(probs_np, 3)} | "
                      f"V={value.item():.2f} | ent={ent_now:.3f} | max_p={max_prob:.3f}")

                if ent_now < 0.5:
                    print(f"  !! ENTROPY WARNING: {ent_now:.4f} — policy collapsing")
                if max_prob > 0.60:
                    print(f"  !! SPAM WARNING: one action at {max_prob:.1%}")

            if episode_steps > 30000:
                print("\nEpisode timeout")
                done = True

            if done:
                controls.release_all()
                if info['lives'][0] > info['lives'][1]:
                    p1_wins += 1
                elif info['lives'][1] > info['lives'][0]:
                    p2_wins += 1

                print(f"\n  Action counts this episode:")
                for i, (name, count) in enumerate(zip(ACTION_NAMES, action_counts)):
                    pct = 100 * count / max(episode_steps, 1)
                    flag = " <-- SPAM" if pct > 40 else ""
                    print(f"    {name:<12} {count:>5}  ({pct:.1f}%){flag}")

                print(f"\n{'='*70}")
                print(f"EPISODE {global_episode} COMPLETE")
                print(f"Reward: {episode_reward:.2f} | Steps: {episode_steps} | "
                      f"Deaths: {deaths_this_ep}")
                print(f"Final: P1={int(info['lives'][0])} lives, "
                      f"P2={int(info['lives'][1])} lives")
                print(f"Win Rate: P1={p1_wins}/{global_episode} "
                      f"({100*p1_wins/global_episode:.1f}%)")
                break

        episode_batch += 1

        if episode_batch >= episodes_per_update:
            if len(memory.combined_data) > 0:
                print(f"\nTraining on {len(memory.combined_data)} experiences "
                      f"({episode_batch} episodes)...")
                loss_val, ent_val = _run_ppo_update(
                    model, optimizer, memory, device,
                    gamma, lam, epsilon, epochs_per_update,
                    entropy_coef, diversity_coef
                )
                print(f"Loss: {loss_val:.4f} | Entropy: {ent_val:.4f}")

                if ent_val < 0.5:
                    print("!! Entropy collapsed — check reward scale / lr")
                elif ent_val > 1.85:
                    print("!! Entropy still near-random — policy not converging")

                memory.clear()
            episode_batch = 0

            ckpt_path = f'checkpoints_14/ppo_ep{global_episode}.pth'
            for path in [ckpt_path, 'checkpoints_14/ppo_latest.pth']:
                torch.save({
                    'episode':              global_episode,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'p1_wins':              p1_wins,
                    'p2_wins':              p2_wins,
                }, path)
            print(f"Checkpoint saved: {ckpt_path}")

        ep_csv = 'logs_14/episodes.csv'
        write_hdr = not os.path.exists(ep_csv) or os.path.getsize(ep_csv) == 0
        with open(ep_csv, 'a') as f:
            if write_hdr:
                count_header = ','.join(f'count_{n}' for n in ACTION_NAMES)
                prob_header  = ','.join(f'prob_{n}' for n in ACTION_NAMES)
                f.write(
                    'timestamp,episode,reward,entropy,loss,steps,deaths,'
                    'p1_lives,p2_lives,p1_wins,p2_wins,'
                    'r_damage_dealt,r_damage_taken,r_time_penalty,'
                    'r_offstage_penalty,r_suicide_penalty,r_death_penalty,r_kill_reward,'
                    f'{count_header},{prob_header}\n'
                )

            rc = env.reward_components
            counts_str = ','.join(str(c) for c in action_counts)
            probs_str  = ','.join(f'{p:.4f}' for p in final_probs) \
                if final_probs is not None else ','.join(['0.0000'] * NUM_ACTIONS)

            f.write(
                f'{time.strftime("%Y-%m-%d %H:%M:%S")},'
                f'{global_episode},{episode_reward:.2f},{ent_val:.6f},{loss_val:.6f},{episode_steps},{deaths_this_ep},'
                f'{int(info["lives"][0])},{int(info["lives"][1])},'
                f'{p1_wins},{p2_wins},'
                f'{rc["damage_dealt"]:.2f},{rc["damage_taken"]:.2f},{rc["time_penalty"]:.2f},'
                f'{rc["offstage_penalty"]:.2f},{rc["suicide_penalty"]:.2f},{rc["death_penalty"]:.2f},{rc["kill_reward"]:.2f},'
                f'{counts_str},{probs_str}\n'
            )


def _run_ppo_update(model, optimizer, memory, device,
                    gamma, lam, epsilon, epochs,
                    entropy_coef, diversity_coef):
    cd_b, actions_b, rewards_b, values_b, \
        old_lp_b, dones_b = memory.get_batches()

    advantages, returns = compute_gae(rewards_b, values_b, dones_b, gamma, lam)
    advantages = np.array(advantages)
    returns    = np.array(returns)

    if advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = advantages - advantages.mean()

    target_dist = TARGET_ACTION_DIST.to(device)

    loss_val = 0.0
    ent_val  = 0.0

    for _ in range(epochs):
        cd_t  = torch.FloatTensor(cd_b).to(device)
        a_t   = torch.LongTensor(actions_b).to(device)
        olp_t = torch.FloatTensor(old_lp_b).to(device)
        adv_t = torch.FloatTensor(advantages).to(device)
        ret_t = torch.FloatTensor(returns).to(device)

        probs, vals = model(cd_t)
        probs = torch.clamp(probs, min=1e-6, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        dist_obj = torch.distributions.Categorical(probs)
        nlp      = dist_obj.log_prob(a_t)
        ent      = dist_obj.entropy().mean()

        ratio = torch.exp(nlp - olp_t)
        s1    = ratio * adv_t
        s2    = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv_t

        actor_loss  = -torch.min(s1, s2).mean()
        critic_loss = nn.MSELoss()(vals.squeeze(), ret_t)

        mean_probs   = probs.mean(dim=0)
        mean_probs   = mean_probs / mean_probs.sum()
        kl_to_target = (target_dist * (target_dist.log() - mean_probs.log())).sum()
        diversity_loss = kl_to_target

        loss = (actor_loss
                + 1.0 * critic_loss
                - entropy_coef * ent
                + diversity_coef * diversity_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loss_val = loss.item()
        ent_val  = ent.item()

    return loss_val, ent_val


if __name__ == "__main__":
    train_ppo()