# Brawlhalla AI Agent

A reinforcement learning agent trained to play **Brawlhalla** using CNN-based visual feature extraction. The agent learns core mechanics, strategy, and split-second decision-making to compete at a competent level — with the goal of a live gameplay demo on demo day.

---

## 🎮 About Brawlhalla

Brawlhalla is a free-to-play platform fighter similar to Super Smash Bros. Players knock opponents off the stage by building up their damage percentage, causing them to fly further when hit.

| Category | Description |
|---|---|
| **Main Goal** | Knock your opponent off the stage more times than they knock you off |
| **Basic Moves** | Walk, jump, dash, light attacks (quick/weak), heavy attacks (slow/strong), and two weapon types |
| **Defense** | Dodge for temporary invincibility, recovery moves to return to stage |
| **Advanced Techniques** | Combos, spacing, and edge-guarding |

---

## 🎥 Presentation

- [Presentation Video](https://drive.google.com/file/d/1EDIkbhnhX2mcYtsJLOkp9HlfbWn-ojaj/view?usp=sharing)

--- 

## 🤖 What the AI Needs to Learn

- Land hits while avoiding damage
- Control the center of the stage
- Execute moves with split-second timing
- Adapt strategies against different opponents and situations

---

## 🧠 Approach

- **Feature Extraction:** CNN processing raw game frames captured directly from the Brawlhalla client (macOS compatible)
- **Training:** Reinforcement learning to teach the agent mechanics, combos, stage control, and adaptive strategy
- **Environment:** Custom OpenAI Gym environment

---

## 🚀 Stretch Goals

- **Character-specific models** — each character has a unique playstyle that may warrant its own trained agent
- **Self-play** — two agents competing against each other on one local device
- **Online ranked play** — deploy an agent into live PvP and benchmark it against real players vs. CPU players

---
# Training Locally

## Prerequisites

- Steam account with Brawlhalla installed
- Python environment with pip

## Installation

### 1. Install Brawlhalla

Download and install Brawlhalla through Steam:

[https://store.steampowered.com/app/291550/Brawlhalla/](https://store.steampowered.com/app/291550/Brawlhalla/)

### 2. Install Python Dependencies

Navigate to the project directory and install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Configure Monitor Settings

Locate `src/global_vars.py` and set the `MONITOR` variable to match the display running Brawlhalla.

## Running the Training Environment

### 1. Launch Brawlhalla

Open Brawlhalla and start a couch party offline game.

### 2. Select a Training Model

Navigate to the `src/run` directory and choose one of the available models, or create your own using the PPOTrainer class:

- `bc_api_input/run.py`
- `cnn_api_input/run.py`
- `eight_val/run.py`
- `YOUR_MODEL_DIRECTORY/run.py`
### 3. Start Training

Once the couch party game is confirmed and loading, execute your selected model:

```bash
python src/run/bc_api_input/run.py
```

or

```bash
python src/run/cnn_api_input/run.py
```

or

```bash
python src/run/eight_val/run.py
```

or

```bash
python src/run/YOUR_MODEL_DIRECTORY/run.py
```

The environment will operate autonomously from this point forward. You may leave the training process running unattended.

## Notes

Ensure that Brawlhalla is actively loading the couch party game before starting the training script to ensure proper initialization.

---
## Future Work

- **Continue Training**: Continue training the agent to see how it performs beyond our current episodes

- **Higher Difficulty CPUs**: Evaluate how the agent performs against harder CPU opponents

- **Reward Function Refinement**: Iterate on the reward system to find the most optimal reward function

- **Multi-Character**: Test the model across different characters, as movement speed, attack damage, and shields differ amongst characters
—

---
## 📚 Related Work

- [ML-Brawlhalla by Tiger767](https://github.com/Tiger767/ML-Brawlhalla?tab=readme-ov-file)

---

## 👥 Collaborators

- Bryan Guen
- Ethan Xin
- Tafari Darosa-Levy
- Praphul Pemmaraju
