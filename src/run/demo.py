import torch
import numpy as np
import cv2
import time
from src.screen_grab.grab import ScreenGrab
from src.health_api.health import HealthAPI
from src.controls.controls import Controls
from src.player_location.player_detector import PlayerDetector

STARTING_LIVES = 5
ACTION_NAMES = ['neutral', 'move_left', 'move_right', 'jump', 'light', 'heavy', 'dodge']


class ActorCritic(torch.nn.Module):
    def __init__(self, input_channels=2, num_actions=7):
        super(ActorCritic, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        conv_out_size = self._get_conv_out((input_channels, 90, 160))
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size + 8, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_actions),
            torch.nn.Softmax(dim=-1)
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size + 8, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, frames, combined_data):
        conv_out = self.conv(frames).view(frames.size()[0], -1)
        data_flat = combined_data.reshape(combined_data.size()[0], -1)
        combined = torch.cat([conv_out, data_flat], dim=1)
        return self.actor(combined), self.critic(combined)


def capture_frame(screen, health_api, player_detector):
    frames = []
    full_frame = None
    for _ in range(2):
        full_frame = screen.grab(greyscale=False)
        game_area  = full_frame[1:1428, 70:2402]
        gray       = cv2.cvtColor(game_area, cv2.COLOR_BGR2GRAY)
        resized    = cv2.resize(gray, (160, 90))
        frames.append(resized / 255.0)

    health_vector, is_player_dead, winner, lives, is_game_over = \
        health_api.process_frame(full_frame)
    location_matrix = player_detector.get_positions()

    normalized_health = health_vector / 100.0
    normalized_lives  = lives / float(STARTING_LIVES)

    location_matrix = np.array(location_matrix, dtype=np.float32)
    location_matrix[:, 0] = (location_matrix[:, 0] - 70)  / (2402 - 70)
    location_matrix[:, 1] = (location_matrix[:, 1] - 1)   / (1428 - 1)
    location_matrix = np.clip(location_matrix, 0, 1)

    scraped_data  = np.stack([normalized_health, normalized_lives], axis=0).T
    combined_data = np.concatenate([scraped_data, location_matrix], axis=1)
    stacked_frames = np.stack(frames, axis=0)
    return stacked_frames, combined_data, is_player_dead, is_game_over


def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = 'checkpoints_9/ppo_ep16.pth'
    model = ActorCritic(input_channels=2, num_actions=7).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    screen          = ScreenGrab(monitor=1)
    health_api      = HealthAPI(starting_lives=STARTING_LIVES)
    player_detector = PlayerDetector(monitor=1)
    controls        = Controls()

    episode = 0

    while True:
        episode += 1
        print(f"\n--- Demo episode {episode} ---")

        health_api.health   = np.array([100.0, 100.0])
        health_api.lives    = np.array([STARTING_LIVES, STARTING_LIVES])
        health_api.last_valid_health_p1 = 100
        health_api.last_valid_health_p2 = 100

        if episode > 1:
            controls.release_all()
            controls.reset_game()
            print("Game reset")

        stacked_frames, combined_data, _, _ = capture_frame(
            screen, health_api, player_detector)

        step         = 0
        action_counts = [0] * 7

        while True:
            s_t  = torch.FloatTensor(stacked_frames).unsqueeze(0).to(device)
            cd_t = torch.FloatTensor(combined_data).unsqueeze(0).to(device)

            with torch.no_grad():
                action_probs, _ = model(s_t, cd_t)
                action_probs = torch.clamp(action_probs, min=1e-6, max=1.0)
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                action = torch.argmax(action_probs, dim=-1)  # greedy — no sampling

            action_idx = action.item()
            action_counts[action_idx] += 1

            controls.execute_action(action_idx)
            time.sleep(0.0089)
            controls.execute_action(action_idx)
            time.sleep(0.0089)

            stacked_frames, combined_data, is_player_dead, is_game_over = \
                capture_frame(screen, health_api, player_detector)

            step += 1

            if step % 100 == 0:
                probs_np = action_probs.cpu().numpy()[0]
                print(f"  [step {step}] action: {ACTION_NAMES[action_idx]} | "
                      f"probs: {np.round(probs_np, 3)}")

            if is_game_over:
                controls.release_all()
                lives = combined_data[:, 1] * float(STARTING_LIVES)
                p1, p2 = int(lives[0]), int(lives[1])
                print(f"Episode {episode} over | P1: {p1} lives, P2: {p2} lives")
                print("Action counts:")
                for name, count in zip(ACTION_NAMES, action_counts):
                    pct = 100 * count / max(step, 1)
                    print(f"  {name:<12} {count:>5}  ({pct:.1f}%)")
                break

            if step > 30000:
                controls.release_all()
                print("Timeout")
                break


if __name__ == "__main__":
    run_demo()