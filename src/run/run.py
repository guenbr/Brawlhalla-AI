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


class ActorCritic(nn.Module):
    def __init__(self, input_channels=2, num_actions=8):  # Changed from 4 to 2
        super(ActorCritic, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out((input_channels, 90, 160))  # Updated to match resized frames

        self.actor = nn.Sequential(
            nn.Linear(conv_out_size + 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(conv_out_size + 4, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, frames, health_data):
        conv_out = self.conv(frames).view(frames.size()[0], -1)
        health_flat = health_data.reshape(health_data.size()[0], -1)
        combined = torch.cat([conv_out, health_flat], dim=1)

        action_probs = self.actor(combined)
        state_value = self.critic(combined)
        return action_probs, state_value


class PPOMemory:
    def __init__(self):
        self.states = []
        self.health_data = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def store(self, state, health, action, reward, value, log_prob, done):
        self.states.append(state)
        self.health_data.append(health)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.health_data = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def get_batches(self):
        return (np.array(self.states),
                np.array(self.health_data),
                np.array(self.actions),
                np.array(self.rewards),
                np.array(self.values),
                np.array(self.log_probs),
                np.array(self.dones))


class BrawlhallaEnv:
    def __init__(self, monitor=0, frame_skip=4, starting_lives=99):
        self.screen = ScreenGrab(monitor=monitor)
        self.health_api = HealthAPI(starting_lives=starting_lives)
        self.controls = Controls()
        self.prev_health = np.array([100, 100])
        self.frame_skip = frame_skip
        self.first_reset = True
        self.recent_actions = deque(maxlen=20)
        self.episode_num = 0
        self.max_episode_time = 480
        self.episode_start_time = None
    def reset(self):
        if self.first_reset:
            print("\n🎮 Starting first episode (no reset)...")
            self.first_reset = False
        else:
            print("\n🎮 Resetting game...")
            self.controls.release_all()
            self.controls.reset_game()

            self.health_api.health = np.array([100, 100])
            self.health_api.lives = np.array([99, 99])
            self.health_api.last_valid_health_p1 = 100
            self.health_api.last_valid_health_p2 = 100

            print("✓ Game reset complete!")

        self.prev_health = np.array([100, 100])
        self.episode_start_time = time.time()
        stacked_frames, health_data, _, _ = self.capture_frame()

        return stacked_frames, health_data

    def capture_frame(self):
        frames = []
        full_frame = None

        for i in range(2):
            full_frame = self.screen.grab(greyscale=False)
            game_area = full_frame[1:1428, 70:2402]
            game_area_gray = cv2.cvtColor(game_area, cv2.COLOR_BGR2GRAY)
            game_area_resized = cv2.resize(game_area_gray, (160, 90))
            frames.append(game_area_resized / 255.0)

        health_vector, is_player_dead, winner, lives, is_game_over = self.health_api.process_frame(full_frame)

        # BACKUP: If lives are 0 but game_over not detected, force it
        if (lives[0] == 0 or lives[1] == 0) and not is_game_over:
            print("⚠️ Forcing game over (lives reached 0)")
            is_game_over = True

        health_data = np.stack([health_vector, lives], axis=0).T
        stacked_frames = np.stack(frames, axis=0)

        return stacked_frames, health_data, is_player_dead, is_game_over
    def step(self, action):
        total_reward = 0

        for _ in range(self.frame_skip):
            self.controls.execute_action(action)
            time.sleep(0.0089)

        stacked_frames, health_data, is_player_dead, is_game_over = self.capture_frame()

        health = health_data[:, 0]

        reward = self.calculate_reward(health, is_player_dead, is_game_over, action)
        total_reward += reward

        if is_player_dead:
            print("💀 Player died - waiting 3 seconds for respawn...")
            time.sleep(3)
            self.health_api.last_valid_health_p1 = 100
            self.health_api.last_valid_health_p2 = 100
            self.prev_health = np.array([100, 100])

        self.prev_health = health.copy()

        info = {
            'health': health,
            'lives': health_data[:, 1],
            'winner': None,
            'is_player_dead': is_player_dead
        }

        return stacked_frames, health_data, total_reward, is_game_over, info

    def calculate_reward(self, health, is_player_dead, is_game_over, action):
        health_diff = health - self.prev_health
        if health_diff[1] < 0:
            reward = abs(health_diff[1]) * 23
        else:
            reward = 0

        if health_diff[0] < 0:
            reward += health_diff[0] * 1.0

        if health_diff[0] > 50:
            reward += health_diff[0] * 8

        reward -= 0.005

        # Stock rewards
        if is_player_dead:
            if health[0] == 0:
                reward -= 1000
            if health[1] == 0:
                reward += 5000

        # EXPLORATION BONUS (first 100 episodes only)
        initial_lives = 99
        lives_lost = initial_lives - health[0]  # How many lives P1 has lost

        if lives_lost < 20:
            self.recent_actions.append(action)
            unique_actions = len(set(self.recent_actions))
            reward += unique_actions * 0.1

        return reward

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def train_ppo():
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available! Training on CPU will be very slow.")
        print("Install PyTorch with CUDA:")
        print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("BRAWLHALLA PPO TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print("=" * 70)

    os.makedirs('checkpoints', exist_ok=True)

    env = BrawlhallaEnv(monitor=1, frame_skip=2, starting_lives=99)

    model = ActorCritic(input_channels=2, num_actions=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # Load from your last checkpoint
    start_episode = 0
    checkpoint_path = 'checkpoints/ppo_episode_19_batch_166.pth'

    if os.path.exists(checkpoint_path):
        print(f"\n📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        deaths_at_checkpoint = checkpoint.get('deaths', 0)
        print(f"✓ Resumed from episode {start_episode}, {deaths_at_checkpoint} deaths")
        print(f"✓ Continuing training...\n")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    memory = PPOMemory()

    gamma = 0.99
    lam = 0.95
    epsilon = 0.2
    epochs_per_episode = 4

    num_episodes = 500

    for episode in range(start_episode, num_episodes):
        state, health_data = env.reset()
        episode_reward = 0
        episode_steps = 0

        deaths_this_episode = 0
        mini_batch_counter = 0

        print(f"\n{'=' * 70}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'=' * 70}")

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            health_tensor = torch.FloatTensor(health_data).unsqueeze(0).to(device)

            with torch.no_grad():
                action_probs, value = model(state_tensor, health_tensor)

            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            next_state, next_health, reward, done, info = env.step(action.item())

            memory.store(state, health_data, action.item(), reward, value.item(), log_prob.item(), done)

            state = next_state
            health_data = next_health
            episode_reward += reward
            episode_steps += 1

            if info['is_player_dead']:
                deaths_this_episode += 1

            if deaths_this_episode > 0 and deaths_this_episode % 90 == 0:
                mini_batch_counter += 1
                print(f"\n🔄 Mini-batch training #{mini_batch_counter} (after {deaths_this_episode} deaths)...")

                if len(memory.states) > 0:
                    states, health_batch, actions, rewards, values, old_log_probs, dones = memory.get_batches()

                    advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
                    advantages = np.array(advantages)
                    returns = np.array(returns)
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    for epoch in range(epochs_per_episode):
                        states_tensor = torch.FloatTensor(states).to(device)
                        health_tensor = torch.FloatTensor(health_batch).to(device)
                        actions_tensor = torch.LongTensor(actions).to(device)
                        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(device)
                        advantages_tensor = torch.FloatTensor(advantages).to(device)
                        returns_tensor = torch.FloatTensor(returns).to(device)

                        action_probs, state_values = model(states_tensor, health_tensor)

                        dist = torch.distributions.Categorical(action_probs)
                        new_log_probs = dist.log_prob(actions_tensor)
                        entropy = dist.entropy().mean()

                        ratio = torch.exp(new_log_probs - old_log_probs_tensor)

                        surr1 = ratio * advantages_tensor
                        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages_tensor
                        actor_loss = -torch.min(surr1, surr2).mean()

                        critic_loss = nn.MSELoss()(state_values.squeeze(), returns_tensor)

                        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()

                    print(f"✓ Mini-batch training complete | Loss: {loss.item():.4f}")
                    memory.clear()

                    checkpoint_path = f'checkpoints/ppo_episode_{episode + 1}_batch_{mini_batch_counter}.pth'
                    torch.save({
                        'episode': episode + 1,
                        'deaths': deaths_this_episode,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}\n")

            if episode_steps > 30000:
                print(f"\n⚠️ Episode timeout at {episode_steps} steps")
                done = True

            if episode_steps % 50 == 0:
                print(f"Step {episode_steps:4d} | Reward: {episode_reward:7.2f} | "
                      f"P1: HP={info['health'][0]:3.0f} Lives={info['lives'][0]:.0f} | "
                      f"P2: HP={info['health'][1]:3.0f} Lives={info['lives'][1]:.0f}")

            if done:
                print(f"\n{'=' * 70}")
                print(f"EPISODE {episode + 1} COMPLETE")
                print(f"{'=' * 70}")
                print(f"Total Reward: {episode_reward:.2f}")
                print(f"Total Steps: {episode_steps}")
                print(f"Total Deaths: {deaths_this_episode}")
                print(f"Final Lives - P1: {info['lives'][0]:.0f}, P2: {info['lives'][1]:.0f}")
                print(f"{'=' * 70}")
                break

        if len(memory.states) > 0:
            print("\n🔄 Final training for episode...")

            states, health_batch, actions, rewards, values, old_log_probs, dones = memory.get_batches()

            advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
            advantages = np.array(advantages)
            returns = np.array(returns)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for epoch in range(epochs_per_episode):
                states_tensor = torch.FloatTensor(states).to(device)
                health_tensor = torch.FloatTensor(health_batch).to(device)
                actions_tensor = torch.LongTensor(actions).to(device)
                old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(device)
                advantages_tensor = torch.FloatTensor(advantages).to(device)
                returns_tensor = torch.FloatTensor(returns).to(device)

                action_probs, state_values = model(states_tensor, health_tensor)

                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions_tensor)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_tensor)

                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages_tensor
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(state_values.squeeze(), returns_tensor)

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            print(f"✓ Final training complete | Loss: {loss.item():.4f}")
            memory.clear()

    final_path = 'checkpoints/ppo_final.pth'
    torch.save({
        'episode': num_episodes,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print(f"Final model saved: {final_path}")
    print("=" * 70)


if __name__ == "__main__":
    train_ppo()