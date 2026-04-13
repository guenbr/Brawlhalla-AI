"""
Behavioral Cloning for Brawlhalla AI
=====================================
Two modes:
  python record_and_pretrain.py record     — play the game, log your actions
  python record_and_pretrain.py pretrain   — train the model on your recorded data
  python record_and_pretrain.py both       — record then immediately pretrain
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import time
import os
from pynput import keyboard

from src.screen_grab.grab import ScreenGrab
from src.health_api.health import HealthAPI
from src.controls.controls import Controls
from src.player_location.player_detector import PlayerDetector

# ── import your model and constants from train.py ────────────────────────────
from run import ActorCritic, BrawlhallaEnv, ACTION_NAMES, NUM_ACTIONS, COMBINED_DATA_SIZE as INPUT_SIZE

STARTING_LIVES  = 15
DATASET_PATH    = 'bc_dataset_boosted_x.pkl'
CHECKPOINT_PATH = 'checkpoints_14/ppo_latest.pth'
BC_EPOCHS       = 5
BC_BATCH_SIZE   = 64
BC_LR           = 0.0003

# ── Keybind map ───────────────────────────────────────────────────────────────
# Edit these to match your actual Brawlhalla keybinds.
# For combo actions (left_heavy = left + heavy key) they are handled below.
#
# Single-key actions:
SINGLE_KEY_MAP = {
    'a': 1,   # move_left
    'd': 2,   # move_right
    'w': 3,   # jump
    'j': 4,   # light
    'k': 5,   # heavy
    'l': 6,   # dodge
}

COMBO_KEY_MAP = {
    frozenset(['a', 'k']): 7,   # left_heavy
    frozenset(['d', 'k']): 8,   # right_heavy
    frozenset(['a', 'j']): 9,   # left_light
    frozenset(['d', 'j']): 10,  # right_light
}

# ── Shared key state ──────────────────────────────────────────────────────────
current_keys  = set()
stop_recording = False


def on_press(key):
    global stop_recording
    try:
        current_keys.add(key.char)
    except AttributeError:
        if key == keyboard.Key.esc:
            stop_recording = True


def on_release(key):
    try:
        current_keys.discard(key.char)
    except AttributeError:
        pass


def get_current_action():
    pressed = frozenset(current_keys)

    # combos first — a+k beats standalone a or k
    for combo, action in COMBO_KEY_MAP.items():
        if combo.issubset(pressed):
            return action

    # then singles
    for key, action in SINGLE_KEY_MAP.items():
        if key in pressed:
            return action

    return 0  # neutral

# ── Recording ─────────────────────────────────────────────────────────────────
def record():
    global stop_recording
    stop_recording = False

    print("=" * 60)
    print("BEHAVIORAL CLONING — RECORDING MODE")
    print("=" * 60)
    print(f"Keybinds:")
    for k, v in SINGLE_KEY_MAP.items():
        print(f"  {k} -> {ACTION_NAMES[v]}")
    for combo, v in COMBO_KEY_MAP.items():
        print(f"  {'+'.join(sorted(combo))} -> {ACTION_NAMES[v]}")
    print()
    print("Press ESC to stop recording.")
    print("=" * 60)

    env      = BrawlhallaEnv(monitor=1, frame_skip=2, starting_lives=STARTING_LIVES)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # load existing dataset so you can record in multiple sessions
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Loaded existing dataset: {len(dataset)} timesteps")
    else:
        dataset = []

    combined_data = env.reset()
    episode       = 0
    steps         = 0

    try:
        while not stop_recording:
            action = get_current_action()

            dataset.append((combined_data.copy(), action))
            steps += 1

            combined_data, _, done, info = env.step(action)

            if steps % 100 == 0:
                print(f"  Recorded {steps} steps | Episode {episode} | "
                      f"P1 lives: {int(info['lives'][0])}  "
                      f"P2 lives: {int(info['lives'][1])}")

            if done:
                episode += 1
                print(f"Episode {episode} done — resetting")
                combined_data = env.reset()

    except Exception as e:
        print(f"Recording stopped: {e}")
    finally:
        listener.stop()
        env.controls.release_all()

        with open(DATASET_PATH, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"\nSaved {len(dataset)} total timesteps to {DATASET_PATH}")

        # print action distribution so you can see if your data is balanced
        actions = [a for _, a in dataset]
        print("\nAction distribution in dataset:")
        for i, name in enumerate(ACTION_NAMES):
            count = actions.count(i)
            pct   = 100 * count / max(len(actions), 1)
            print(f"  {name:<14} {count:>6}  ({pct:.1f}%)")


# ── Pretraining ───────────────────────────────────────────────────────────────
def pretrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if not os.path.exists(DATASET_PATH):
        print(f"No dataset found at {DATASET_PATH} — run record mode first.")
        return

    with open(DATASET_PATH, 'rb') as f:
        dataset = pickle.load(f)

    dataset = [(cd, a) for (cd, a) in dataset if a != 0]

    print(f"Loaded {len(dataset)} timesteps for pretraining")

    cds, actions = zip(*dataset)
    cds     = torch.FloatTensor(np.array(cds)).to(device)
    actions = torch.LongTensor(np.array(actions)).to(device)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(cds, actions),
        batch_size=BC_BATCH_SIZE,
        shuffle=True
    )

    os.makedirs('checkpoints_13', exist_ok=True)

    model     = ActorCritic(input_size=INPUT_SIZE, num_actions=NUM_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=BC_LR)

    # load existing checkpoint if present so BC fine-tunes rather than resets
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading existing checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print("Checkpoint loaded — BC will fine-tune from here")
    else:
        print("No checkpoint found — BC will train from scratch")

    weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 5.0, 5.0, 3.0, 5.0, 5.0, 4.0, 4.0])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    print(f"\nPretraining for {BC_EPOCHS} epochs...")
    print("=" * 60)

    for epoch in range(BC_EPOCHS):
        total_loss   = 0.0
        correct      = 0
        total        = 0

        for cd_batch, a_batch in loader:
            probs, _ = model(cd_batch)
            loss     = criterion(probs, a_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted   = probs.argmax(dim=1)
            correct    += (predicted == a_batch).sum().item()
            total      += a_batch.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1:>3}/{BC_EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Accuracy: {accuracy:.1f}%")

    # save as a PPO-compatible checkpoint
    torch.save({
        'episode':              0,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'p1_wins':              0,
        'p2_wins':              0,
    }, CHECKPOINT_PATH)

    print(f"\nBC checkpoint saved to {CHECKPOINT_PATH}")
    print("You can now run train.py — it will load this as the starting policy.")

    # print what the policy now prefers
    print("\nFinal policy action preferences (averaged over dataset):")
    model.eval()
    with torch.no_grad():
        probs, _ = model(cds)
        mean_probs = probs.mean(dim=0).cpu().numpy()
    for i, (name, p) in enumerate(zip(ACTION_NAMES, mean_probs)):
        bar = '█' * int(p * 100)
        print(f"  {name:<14} {p:.3f}  {bar}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else 'record'

    pretrain()