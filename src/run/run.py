import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from src.screen_grab.grab import ScreenGrab
from src.health_api.health import HealthAPI
from src.controls.controls import Controls
import cv2
import time
import os
from src.player_location.player_detector import PlayerDetector

STARTING_LIVES = 5

ACTION_NAMES = ['neutral', 'move_left', 'move_right', 'jump', 'light', 'heavy', 'dodge']


class ActorCritic(nn.Module):
    def __init__(self, input_channels=2, num_actions=7):
        super(ActorCritic, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out((input_channels, 90, 160))

        self.actor = nn.Sequential(
            nn.Linear(conv_out_size + 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(conv_out_size + 8, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, frames, combined_data):
        conv_out = self.conv(frames).view(frames.size()[0], -1)
        data_flat = combined_data.reshape(combined_data.size()[0], -1)
        combined = torch.cat([conv_out, data_flat], dim=1)
        return self.actor(combined), self.critic(combined)


class PPOMemory:
    def __init__(self):
        self.states        = []
        self.combined_data = []
        self.actions       = []
        self.rewards       = []
        self.values        = []
        self.log_probs     = []
        self.dones         = []

    def store(self, state, combined_data, action, reward, value, log_prob, done):
        self.states.append(state)
        self.combined_data.append(combined_data)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.states = []; self.combined_data = []; self.actions = []
        self.rewards = []; self.values = []; self.log_probs = []; self.dones = []

    def get_batches(self):
        return (np.array(self.states),
                np.array(self.combined_data),
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
        self.prev_combined_data = np.zeros((2, 4), dtype=np.float32)

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
        stacked_frames, combined_data, _, _ = self.capture_frame()
        self.prev_combined_data = combined_data.copy()
        return stacked_frames, combined_data

    def capture_frame(self):
        frames = []
        full_frame = None
        for _ in range(2):
            full_frame = self.screen.grab(greyscale=False)
            game_area  = full_frame[1:1428, 70:2402]
            gray       = cv2.cvtColor(game_area, cv2.COLOR_BGR2GRAY)
            resized    = cv2.resize(gray, (160, 90))
            frames.append(resized / 255.0)

        health_vector, is_player_dead, winner, lives, is_game_over = \
            self.health_api.process_frame(full_frame)
        location_matrix = self.player_detector.get_positions()

        normalized_health = health_vector / 100.0
        normalized_lives  = lives / float(self.starting_lives)

        location_matrix = np.array(location_matrix, dtype=np.float32)
        location_matrix[:, 0] = (location_matrix[:, 0] - 70)  / (2402 - 70)
        location_matrix[:, 1] = (location_matrix[:, 1] - 1)   / (1428 - 1)
        location_matrix = np.clip(location_matrix, 0, 1)

        if (lives[0] <= 0 or lives[1] <= 0) and not is_game_over:
            print("Forcing game over (a player's lives reached 0)")
            is_game_over = True

        scraped_data   = np.stack([normalized_health, normalized_lives], axis=0).T
        combined_data  = np.concatenate([scraped_data, location_matrix], axis=1)
        stacked_frames = np.stack(frames, axis=0)
        return stacked_frames, combined_data, is_player_dead, is_game_over

    def step(self, action):
        total_reward = 0

        for _ in range(self.frame_skip):
            self.controls.execute_action(action)
            time.sleep(0.0089)

        stacked_frames, combined_data, is_player_dead, is_game_over = \
            self.capture_frame()

        health = combined_data[:, 0] * 100.0
        lives  = combined_data[:, 1] * float(self.starting_lives)

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
        return stacked_frames, combined_data, total_reward, is_game_over, info

    # -------------------------------------------------------- calculate_reward
    def calculate_reward(self, health, lives, is_player_dead, is_game_over, combined_data, action):
        reward = 0
        health_diff = health - self.prev_health
        damage_dealt = abs(health_diff[1]) if health_diff[1] < 0 else 0
        damage_taken = abs(health_diff[0]) if health_diff[0] < 0 else 0

        p1_x = combined_data[0, 2]
        p2_x = combined_data[1, 2]
        dist = abs(p1_x - p2_x)

        # ── 1. DAMAGE — multiplied by proximity so movement pays off ──
        proximity_bonus = max(1.0, 2.0 - dist * 4.0)
        reward += damage_dealt * 2.0 * proximity_bonus
        reward -= damage_taken * 0.8

        # ── 2. PLATFORM ───────────────────────────────────────────────
        if 0.28 < p1_x < 0.80:
            reward += 0.05
        else:
            reward -= 0.3

        # ── 3. CLOSING DISTANCE ───────────────────────────────────────
        prev_dist = abs(self.prev_combined_data[0, 2] - self.prev_combined_data[1, 2])
        curr_dist = dist
        closing = prev_dist - curr_dist
        reward += closing * 6.0

        # ── 4. DIVERSITY ──────────────────────────────────────────────
        self.recent_actions.append(action)
        if len(self.recent_actions) >= 10:
            unique = len(set(list(self.recent_actions)[-10:]))
            if unique <= 2:
                reward -= 0.15
            elif unique >= 4:
                reward += 0.05

        # ── 5. STILLNESS ──────────────────────────────────────────────
        p1_dx = abs(p1_x - self.prev_combined_data[0, 2])
        if p1_dx < 0.002:
            reward -= 0.35
        elif p1_dx > 0.005:
            reward += 0.15

        # ── 6. DEATH OUTCOMES ─────────────────────────────────────────
        if is_player_dead:
            if health[0] <= 1:
                reward -= 10
                print(f"  P1 DIED | Lives: {int(lives[0])}")
            if health[1] <= 1:
                reward += 20
                print(f"  P1 GOT A KILL | Opponent lives: {int(lives[1])}")

        # ── 7. EPISODE OUTCOME ────────────────────────────────────────
        if is_game_over:
            p1, p2 = int(lives[0]), int(lives[1])
            if p1 > p2:
                reward += 20 + (p1 - p2) * 2
                print(f"  EPISODE WIN  | P1: {p1} P2: {p2}")
            elif p2 > p1:
                reward -= 20
                print(f"  EPISODE LOSS | P1: {p1} P2: {p2}")
            else:
                print(f"  EPISODE DRAW | Both: {p1}")

        # ── STORE STATE ───────────────────────────────────────────────
        self.prev_combined_data = combined_data.copy()

        return reward

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
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

    os.makedirs('checkpoints_9', exist_ok=True)
    os.makedirs('logs_9', exist_ok=True)

    env       = BrawlhallaEnv(monitor=1, frame_skip=2, starting_lives=STARTING_LIVES)
    model     = ActorCritic(input_channels=2, num_actions=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00015)  # was 0.0003
    controls  = Controls()
    memory    = PPOMemory()

    p1_wins        = 0
    p2_wins        = 0
    episode_offset = 0

    # ── resume from checkpoint ────────────────────────────────────────
    checkpoint_path = 'checkpoints_9/ppo_ep460.pth'  # resume from best checkpoint

    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        for group in optimizer.param_groups:
            group['lr'] = 0.00015

        p1_wins        = ckpt.get('p1_wins', 0)
        p2_wins        = ckpt.get('p2_wins', 0)
        episode_offset = ckpt.get('episode', 0)
        print(f"Resumed from episode {episode_offset} | "
              f"P1 wins: {p1_wins} | P2 wins: {p2_wins}\n")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting fresh\n")

    # ── hyperparameters ───────────────────────────────────────────────
    gamma              = 0.97
    lam                = 0.95
    epsilon            = 0.10   # was 0.15 — tighter clip, slower policy shift
    epochs_per_episode = 1      # was 2
    entropy_coef       = 0.06   # was 0.08
    episodes_per_update = 4     # accumulate before updating
    num_episodes       = 500

    episode_batch = 0
    loss_val      = 0.0
    ent_val       = 0.0

    for episode in range(num_episodes):
        global_episode = episode_offset + episode + 1

        state, combined_data = env.reset()
        episode_reward = 0
        episode_steps  = 0
        deaths_this_ep = 0
        action_counts  = [0] * 7
        final_probs    = None

        print(f"\n{'='*70}")
        print(f"EPISODE {global_episode} (session ep {episode+1}/{num_episodes})  |  "
              f"entropy_coef={entropy_coef:.4f}  |  lives={STARTING_LIVES}")
        print(f"{'='*70}")

        while True:
            s_t  = torch.FloatTensor(state).unsqueeze(0).to(device)
            cd_t = torch.FloatTensor(combined_data).unsqueeze(0).to(device)

            with torch.no_grad():
                action_probs, value = model(s_t, cd_t)
                action_probs = torch.clamp(action_probs, min=1e-6, max=1.0)
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

                # uniform prior — prevents any action probability reaching zero
                #action_probs = 0.95 * action_probs + 0.05 * (1.0 / 7)

                dist     = torch.distributions.Categorical(action_probs)
                action   = dist.sample()
                log_prob = dist.log_prob(action)

            action_counts[action.item()] += 1
            final_probs = action_probs.cpu().numpy()[0]

            next_state, next_cd, reward, done, info = env.step(action.item())
            memory.store(state, combined_data, action.item(), reward,
                         value.item(), log_prob.item(), done)

            state         = next_state
            combined_data = next_cd
            episode_reward += reward
            episode_steps  += 1

            if info['is_player_dead']:
                deaths_this_ep += 1

            if episode_steps % 50 == 0:
                probs_np = action_probs.cpu().numpy()[0]
                ent_now  = -np.sum(probs_np * np.log(probs_np + 1e-8))
                print(f"  [step {episode_steps}] probs: {np.round(probs_np, 3)} | "
                      f"V={value.item():.2f} | ent={ent_now:.3f}")

                if ent_now < 0.3:
                    print(f"  !! ENTROPY WARNING: {ent_now:.4f} — consider stopping")

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
                    print(f"    {name:<12} {count:>5}  ({pct:.1f}%)")

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

        # ── update every N episodes ───────────────────────────────────
        if episode_batch >= episodes_per_update:
            if len(memory.states) > 0:
                print(f"\nTraining on {len(memory.states)} experiences "
                      f"({episode_batch} episodes)...")
                loss_val, ent_val = _run_ppo_update(
                    model, optimizer, memory, device,
                    gamma, lam, epsilon, epochs_per_episode, entropy_coef
                )
                print(f"Loss: {loss_val:.4f} | Entropy: {ent_val:.4f}")

                if ent_val < 0.3:
                    print("!! Entropy collapsed — check your reward scale / lr")

                memory.clear()
            episode_batch = 0

            # save checkpoint after each update
            ckpt_path = f'checkpoints_9/ppo_ep{global_episode}.pth'
            torch.save({
                'episode':              global_episode,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'p1_wins':              p1_wins,
                'p2_wins':              p2_wins,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        # ── csv log every episode ─────────────────────────────────────
        csv_path  = 'logs_9/training_metrics.csv'
        write_hdr = not os.path.exists(csv_path) or \
                    os.path.getsize(csv_path) == 0
        with open(csv_path, 'a') as f:
            if write_hdr:
                count_header = ','.join(f'count_{n}' for n in ACTION_NAMES)
                prob_header  = ','.join(f'prob_{n}'  for n in ACTION_NAMES)
                f.write(
                    'timestamp,episode,deaths,reward,steps,'
                    'p1_lives,p2_lives,p1_wins,p2_wins,entropy,'
                    f'{count_header},{prob_header}\n'
                )

            counts_str = ','.join(str(c) for c in action_counts)
            probs_str  = ','.join(f'{p:.4f}' for p in final_probs) \
                         if final_probs is not None else ','.join(['0.0000'] * 7)

            f.write(
                f'{time.strftime("%Y-%m-%d %H:%M:%S")},'
                f'{global_episode},{deaths_this_ep},{episode_reward:.2f},'
                f'{episode_steps},{int(info["lives"][0])},'
                f'{int(info["lives"][1])},{p1_wins},{p2_wins},'
                f'{ent_val:.4f},{counts_str},{probs_str}\n'
            )


def _run_ppo_update(model, optimizer, memory, device,
                    gamma, lam, epsilon, epochs, entropy_coef):
    states_b, cd_b, actions_b, rewards_b, values_b, \
        old_lp_b, dones_b = memory.get_batches()

    advantages, returns = compute_gae(rewards_b, values_b, dones_b, gamma, lam)
    advantages = np.array(advantages)
    returns    = np.array(returns)

    if advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = advantages - advantages.mean()

    loss_val = 0
    ent_val  = 0

    for _ in range(epochs):
        s_t   = torch.FloatTensor(states_b).to(device)
        cd_t  = torch.FloatTensor(cd_b).to(device)
        a_t   = torch.LongTensor(actions_b).to(device)
        olp_t = torch.FloatTensor(old_lp_b).to(device)
        adv_t = torch.FloatTensor(advantages).to(device)
        ret_t = torch.FloatTensor(returns).to(device)

        probs, vals = model(s_t, cd_t)
        probs = torch.clamp(probs, min=1e-6, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # uniform prior during update too
        #probs = 0.95 * probs + 0.05 * (1.0 / 7)

        dist = torch.distributions.Categorical(probs)
        nlp  = dist.log_prob(a_t)
        ent  = dist.entropy().mean()

        ratio = torch.exp(nlp - olp_t)
        s1    = ratio * adv_t
        s2    = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv_t

        actor_loss  = -torch.min(s1, s2).mean()
        critic_loss = nn.MSELoss()(vals.squeeze(), ret_t)

        # stronger entropy bonus when entropy drops low
        entropy_weight = entropy_coef
        loss = actor_loss + 1.0 * critic_loss - entropy_weight * ent

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loss_val = loss.item()
        ent_val  = ent.item()

    return loss_val, ent_val


if __name__ == "__main__":
    train_ppo()