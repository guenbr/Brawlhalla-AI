import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from pynput import keyboard

from run.train import ActorCritic, BrawlhallaEnv, NUM_ACTIONS

STARTING_LIVES  = 15
DATASET_PATH    = 'bc_dataset_boosted_x.pkl'
CHECKPOINT_PATH = 'checkpoints_14/ppo_latest.pth'
BC_EPOCHS       = 5
BC_BATCH_SIZE   = 64
BC_LR           = 0.0003

SINGLE_KEY_MAP = {
    'a': 1,
    'd': 2,
    'w': 3,
    'j': 4,
    'k': 5,
    'l': 6,
}

COMBO_KEY_MAP = {
    frozenset(['a', 'k']): 7,
    frozenset(['d', 'k']): 8,
    frozenset(['a', 'j']): 9,
    frozenset(['d', 'j']): 10,
}

current_keys  = set()
stop_recording = False


def on_press(key) -> None:
    """
    Key press listener

    Args:
        key (str): current key pressed
    """
    global stop_recording
    try:
        current_keys.add(key.char)
    except AttributeError:
        if key == keyboard.Key.esc:
            stop_recording = True


def on_release(key) -> None:
    """
    Key release listener

    Args:
        key (str): current key released
    """
    try:
        current_keys.discard(key.char)
    except AttributeError:
        pass


def get_current_action() -> str | int:
    """
    Gets current key stroke/strokes
    """
    pressed = frozenset(current_keys)

    # combos first
    for combo, action in COMBO_KEY_MAP.items():
        if combo.issubset(pressed):
            return action

    # then singles
    for key, action in SINGLE_KEY_MAP.items():
        if key in pressed:
            return action

    return 0

# ── Recording ─────────────────────────────────────────────────────────────────
def record() -> None:
    """
    Records human gameplay by listening to keyboard inputs and saving
    (state, action) pairs to a dataset file for behavioral cloning pretraining
    """
    global stop_recording
    stop_recording = False

    env      = BrawlhallaEnv(monitor=1, frame_skip=2, starting_lives=STARTING_LIVES)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Load existing dataset
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = []

    combined_data = env.reset()
    episode       = 0
    steps         = 0

    try:
        while not stop_recording:
            # Record state/action pairs
            action = get_current_action()

            dataset.append((combined_data.copy(), action))
            steps += 1
            combined_data, _, done, info = env.step(action)

            if done:
                episode += 1
                combined_data = env.reset()
    finally:
        listener.stop()
        env.controls.release_all()

        # Save to pkl file
        with open(DATASET_PATH, 'wb') as f:
            pickle.dump(dataset, f)

def pretrain() -> None:
    """
    Runs Behavioral Cloning loop, sets up PPO weights for training
    """
    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    with open(DATASET_PATH, 'rb') as f:
        dataset = pickle.load(f)

    # Get non neutral (state, action) pairs, load to Tensors
    dataset = [(cd, a) for (cd, a) in dataset if a != 0]

    cds, actions = zip(*dataset)
    cds = torch.FloatTensor(np.array(cds)).to(device)
    actions = torch.LongTensor(np.array(actions)).to(device)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(cds, actions),
        batch_size=BC_BATCH_SIZE,
        shuffle=True
    )

    # Init model and optimizer
    os.makedirs('checkpoints_14', exist_ok=True)

    model     = ActorCritic(input_size=14, num_actions=NUM_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=BC_LR)

    # Load previous saved BC checkpoint if exists
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 5.0, 5.0, 3.0, 5.0, 5.0, 4.0, 4.0])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # Run BC loop
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


    # Save as checkpoint
    torch.save({
        'episode':              0,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'p1_wins':              0,
        'p2_wins':              0,
    }, CHECKPOINT_PATH)

    model.eval()
    with torch.no_grad():
        probs, _ = model(cds)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else 'record'
    pretrain()