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

## 📚 Related Work

- [ML-Brawlhalla by Tiger767](https://github.com/Tiger767/ML-Brawlhalla?tab=readme-ov-file)

---

## 👥 Collaborators

- Bryan Guen
- Ethan Xin
- Tafari Darosa-Levy
- Praphul Pemmaraju