import torch
import torch.optim as optim
import numpy as np
from src.controls.controls import Controls
import time
import os
from src.run.brawhalla_env import BrawlhallaEnv
from src.run.ppo_actor_critic import ActorCritic
from src.run.ppo_memory import PPOMemory
from global_vars import MONITOR
from src.run.ppo_functions import _run_ppo_update

ACTION_NAMES = [
    'neutral', 'move_left', 'move_right', 'jump',
    'light', 'heavy', 'dodge', 'left_heavy',
    'right_heavy', 'left_light', 'right_light',
]

NUM_ACTIONS = len(ACTION_NAMES)
COMBINED_DATA_SIZE = 14


class PPOTrainer:
    def __init__(
            self,
            # Training hyperparameters
            gamma=0.965,
            lam=0.95,
            epsilon=0.20,
            epochs_per_update=4,
            entropy_coef=0.003,
            diversity_coef=0.0,
            episodes_per_update=4,
            num_episodes=500,
            learning_rate=0.00015,

            # Environment settings
            frame_skip=2,
            max_steps=30000,
            edge_threshold=0.15,
            edge_mask=True,
            starting_lives=15,
            use_cnn=False,

            # Logging settings
            log_every_n_steps=50,
            checkpoint_dir='checkpoints_14',
            log_dir='logs_14',

            # Warning thresholds
            entropy_warning=0.5,
            entropy_random=1.85,
            max_prob_spam=0.60,
            action_spam_pct=40,
    ):
        # Store all config
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.epochs_per_update = epochs_per_update
        self.entropy_coef = entropy_coef
        self.diversity_coef = diversity_coef
        self.episodes_per_update = episodes_per_update
        self.num_episodes = num_episodes
        self.learning_rate = learning_rate

        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.edge_threshold = edge_threshold
        self.edge_mask = edge_mask
        self.use_cnn = use_cnn
        self.log_every_n_steps = log_every_n_steps
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.entropy_warning = entropy_warning
        self.entropy_random = entropy_random
        self.max_prob_spam = max_prob_spam
        self.action_spam_pct = action_spam_pct

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 70)

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize components
        self.env = BrawlhallaEnv(starting_lives=starting_lives, monitor=MONITOR, frame_skip=frame_skip)
        if use_cnn:
            self.model = ActorCritic(
                num_actions=NUM_ACTIONS,
                use_cnn=True,
                input_channels=2,
                combined_data_size=8
            ).to(self.device)
        else:
            self.model = ActorCritic(
                num_actions=NUM_ACTIONS,
                use_cnn=False,
                input_size=COMBINED_DATA_SIZE
            ).to(self.device)

        self.memory = PPOMemory(use_cnn=use_cnn)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.controls = Controls()

        # Training state
        self.p1_wins = 0
        self.p2_wins = 0
        self.episode_offset = 0

        # Load checkpoint if exists
        self.load_checkpoint()

    def load_checkpoint(self):
        """Load checkpoint if it exists"""
        checkpoint_path = f'{self.checkpoint_dir}/ppo_latest.pth'
        if os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt['model_state_dict'])

            for group in self.optimizer.param_groups:
                group['lr'] = self.learning_rate

            self.p1_wins = ckpt.get('p1_wins', 0)
            self.p2_wins = ckpt.get('p2_wins', 0)
            self.episode_offset = ckpt.get('episode', 0)
            print(f"Resumed from episode {self.episode_offset} | "
                  f"P1 wins: {self.p1_wins} | P2 wins: {self.p2_wins}\n")
        else:
            print("Starting fresh\n")

    def apply_edge_mask(self, action_probs, combined_data):
        """Apply edge masking to action probabilities"""
        if not self.edge_mask:
            return action_probs

        dist_left = combined_data[11]
        dist_right = combined_data[12]

        mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        if dist_right < self.edge_threshold:
            mask[2] = 0  # move_right
            mask[5] = 0  # heavy
            mask[8] = 0  # right_heavy
            mask[10] = 0  # right_light

        if dist_left < self.edge_threshold:
            mask[1] = 0  # move_left
            mask[5] = 0  # heavy
            mask[7] = 0  # left_heavy
            mask[9] = 0  # left_light

        mask_t = torch.FloatTensor(mask).to(self.device)
        action_probs = action_probs * mask_t
        action_probs = action_probs + 1e-6

        return action_probs

    def select_action(self, *args):
        """Select action based on current policy"""
        # Parse inputs based on mode
        if self.use_cnn:
            state, combined_data = args
            s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            cd_t = torch.FloatTensor(combined_data).unsqueeze(0).to(self.device)
        else:
            combined_data = args[0]
            cd_t = torch.FloatTensor(combined_data).unsqueeze(0).to(self.device)

        # Common forward pass and action selection
        with torch.no_grad():
            # Forward pass (conditional on inputs)
            if self.use_cnn:
                action_probs, value = self.model(s_t, cd_t)
            else:
                action_probs, value = self.model(cd_t)

            # Process probabilities (same for both modes)
            action_probs = torch.clamp(action_probs, min=1e-6, max=1.0)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

            action_probs = self.apply_edge_mask(action_probs, combined_data)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item(), action_probs.cpu().numpy()[0]

    def log_step(self, episode_steps, action_probs, value):
        """Log step information"""
        probs_np = action_probs.cpu().numpy()[0]
        ent_now = -np.sum(probs_np * np.log(probs_np + 1e-8))
        max_prob = probs_np.max()

        print(f"  [step {episode_steps}] probs: {np.round(probs_np, 3)} | "
              f"V={value:.2f} | ent={ent_now:.3f} | max_p={max_prob:.3f}")

        if ent_now < self.entropy_warning:
            print(f"  !! ENTROPY WARNING: {ent_now:.4f} — policy collapsing")
        if max_prob > self.max_prob_spam:
            print(f"  !! SPAM WARNING: one action at {max_prob:.1%}")

    def print_episode_summary(self, global_episode, episode_reward, episode_steps,
                              deaths_this_ep, info, action_counts):
        """Print episode summary"""
        print(f"\n  Action counts this episode:")
        for name, count in zip(ACTION_NAMES, action_counts):
            pct = 100 * count / max(episode_steps, 1)
            flag = " <-- SPAM" if pct > self.action_spam_pct else ""
            print(f"    {name:<12} {count:>5}  ({pct:.1f}%){flag}")

        print(f"\n{'=' * 70}")
        print(f"EPISODE {global_episode} COMPLETE")
        print(f"Reward: {episode_reward:.2f} | Steps: {episode_steps} | Deaths: {deaths_this_ep}")
        print(f"Final: P1={int(info['lives'][0])} lives, P2={int(info['lives'][1])} lives")
        print(f"Win Rate: P1={self.p1_wins}/{global_episode} "
              f"({100 * self.p1_wins / global_episode:.1f}%)")

    def update_model(self):
        """Run PPO update"""
        print(f"\nTraining on {len(self.memory.combined_data)} experiences...")
        loss_val, ent_val = _run_ppo_update(
            self.model, self.optimizer, self.memory, self.device,
            self.gamma, self.lam, self.epsilon, self.epochs_per_update,
            self.entropy_coef, self.diversity_coef
        )
        print(f"Loss: {loss_val:.4f} | Entropy: {ent_val:.4f}")

        if ent_val < self.entropy_warning:
            print("!! Entropy collapsed — check reward scale / lr")
        elif ent_val > self.entropy_random:
            print("!! Entropy still near-random — policy not converging")

        return loss_val, ent_val

    def save_checkpoint(self, global_episode):
        """Save model checkpoint"""
        ckpt_path = f'{self.checkpoint_dir}/ppo_ep{global_episode}.pth'
        for path in [ckpt_path, f'{self.checkpoint_dir}/ppo_latest.pth']:
            torch.save({
                'episode': global_episode,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'p1_wins': self.p1_wins,
                'p2_wins': self.p2_wins,
            }, path)
        print(f"Checkpoint saved: {ckpt_path}")

    def log_to_csv(self, global_episode, episode_reward, episode_steps, deaths_this_ep,
                   info, action_counts, final_probs, loss_val, ent_val):
        """Log episode data to CSV"""
        ep_csv = f'{self.log_dir}/episodes.csv'
        write_hdr = not os.path.exists(ep_csv) or os.path.getsize(ep_csv) == 0

        with open(ep_csv, 'a') as f:
            if write_hdr:
                count_header = ','.join(f'count_{n}' for n in ACTION_NAMES)
                prob_header = ','.join(f'prob_{n}' for n in ACTION_NAMES)
                f.write(
                    'timestamp,episode,reward,entropy,loss,steps,deaths,'
                    'p1_lives,p2_lives,p1_wins,p2_wins,'
                    'r_damage_dealt,r_damage_taken,r_time_penalty,'
                    'r_offstage_penalty,r_suicide_penalty,r_death_penalty,r_kill_reward,'
                    f'{count_header},{prob_header}\n'
                )

            rc = self.env.reward_components
            counts_str = ','.join(str(c) for c in action_counts)
            probs_str = ','.join(f'{p:.4f}' for p in final_probs) \
                if final_probs is not None else ','.join(['0.0000'] * NUM_ACTIONS)

            f.write(
                f'{time.strftime("%Y-%m-%d %H:%M:%S")},'
                f'{global_episode},{episode_reward:.2f},{ent_val:.6f},{loss_val:.6f},'
                f'{episode_steps},{deaths_this_ep},'
                f'{int(info["lives"][0])},{int(info["lives"][1])},'
                f'{self.p1_wins},{self.p2_wins},'
                f'{rc["damage_dealt"]:.2f},{rc["damage_taken"]:.2f},{rc["time_penalty"]:.2f},'
                f'{rc["offstage_penalty"]:.2f},{rc["suicide_penalty"]:.2f},'
                f'{rc["death_penalty"]:.2f},{rc["kill_reward"]:.2f},'
                f'{counts_str},{probs_str}\n'
            )

    def run_episode(self):
        """Run a single episode"""
        combined_data = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        deaths_this_ep = 0
        action_counts = [0] * NUM_ACTIONS
        final_probs = None

        while True:
            action, log_prob, value, probs = self.select_action(combined_data)
            action_counts[action] += 1
            final_probs = probs

            next_cd, reward, done, info = self.env.step(action)
            self.memory.store(combined_data, action, reward, value, log_prob, done)

            combined_data = next_cd
            episode_reward += reward
            episode_steps += 1

            if info['is_player_dead']:
                deaths_this_ep += 1

            if episode_steps % self.log_every_n_steps == 0:
                self.log_step(episode_steps, torch.FloatTensor(probs), value)

            if episode_steps > self.max_steps:
                print("\nEpisode timeout")
                done = True

            if done:
                self.controls.release_all()
                if info['lives'][0] > info['lives'][1]:
                    self.p1_wins += 1
                elif info['lives'][1] > info['lives'][0]:
                    self.p2_wins += 1
                break

        return episode_reward, episode_steps, deaths_this_ep, info, action_counts, final_probs

    def train(self):
        """Main training loop"""
        episode_batch = 0
        loss_val = 0.0
        ent_val = 0.0

        for episode in range(self.num_episodes):
            global_episode = self.episode_offset + episode + 1

            # Run episode
            episode_reward, episode_steps, deaths_this_ep, info, action_counts, final_probs = \
                self.run_episode()

            self.print_episode_summary(global_episode, episode_reward, episode_steps,
                                       deaths_this_ep, info, action_counts)

            episode_batch += 1

            # Update model
            if episode_batch >= self.episodes_per_update:
                if len(self.memory.combined_data) > 0:
                    loss_val, ent_val = self.update_model()
                    self.memory.clear()

                episode_batch = 0
                self.save_checkpoint(global_episode)

            # Log to CSV
            self.log_to_csv(global_episode, episode_reward, episode_steps, deaths_this_ep,
                            info, action_counts, final_probs, loss_val, ent_val)


if __name__ == "__main__":
    # Default run
    trainer = PPOTrainer()
    trainer.train()

    # Or customize:
    # trainer = PPOTrainer(learning_rate=0.0003, entropy_coef=0.01, num_episodes=1000)
    # trainer.train()

    # Or run multiple experiments:
    # trainer = PPOTrainer(
    #     learning_rate=0.0005,
    #     checkpoint_dir='checkpoints_fast',
    #     log_dir='logs_fast'
    # )
    # trainer.train()