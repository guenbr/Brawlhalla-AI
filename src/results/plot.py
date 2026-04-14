import pandas as pd
import matplotlib.pyplot as plt

# Load CSV into DataFrame
df = pd.read_csv("data/episodes.csv")


def plot_reward_over_episode():
    x_col ='episode'
    y_col = 'reward'

    window = 5  # adjust rolling window size as needed
    df["moving_avg"] = df[y_col].rolling(window=window).mean()

    plt.plot(df[x_col], df[y_col], label="Raw", alpha=0.6)
    plt.plot(df[x_col], df["moving_avg"], linewidth=2, label=f"{window}-episode MA")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Reward over Episodes")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig("plots/reward_over_episode.png", dpi=150)
    plt.show()


def plot_kd():
    df['kd'] = (15 - df['p2_lives']) / 15
    window = 5  # adjust rolling window as needed
    df['kd_ma'] = df['kd'].rolling(window=window).mean()
    plt.plot(df.index, df['kd'], alpha=0.6, label="KD")
    plt.plot(df.index, df['kd_ma'], linewidth=2, label=f"{window}-period MA")
    plt.xlabel("Episode")
    plt.ylabel("K:D")
    plt.title("K:D over Episodes")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/kd_over_episode.png", dpi=150)
    plt.show()


def plot_action_dist():
    groups = {
        'movement': ['move_left', 'move_right', 'jump', 'neutral'],
        'attacks': ['light', 'heavy'],
        'movement_attack': ['left_light', 'right_light', 'left_heavy', 'right_heavy'],
        'dodge': ['dodge']
    }

    df_sampled = df.iloc[::15]

    for group, actions in groups.items():
        cols = [f'prob_{a}' for a in actions]
        group_prob = df_sampled[cols].sum(axis=1)
        plt.plot(df_sampled.index, group_prob, marker='o', markersize=3, linewidth=1.5, label=group)

    plt.xlabel("Episode")
    plt.ylabel("Probability")
    plt.title("Action Distribution over Episodes")
    plt.legend(loc='upper left', fontsize=5.4)
    plt.grid(linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plots/action_dist.png", dpi=150)
    plt.show()


def plot_entropy_over_episode():
    x_col = 'episode'
    y_col = 'entropy'

    df[y_col] = df[y_col].replace(0, pd.NA).ffill().bfill()

    df_sampled = df.iloc[::1]

    plt.plot(df_sampled[x_col], df_sampled[y_col], alpha=0.6, label="Entropy")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Entropy over Episodes")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig("plots/ntropy_over_episode.png", dpi=150)
    plt.show()

def plot_entropy_reward_normalized():
    x_col = 'episode'

    df['entropy'] = df['entropy'].replace(0, pd.NA).ffill().bfill()
    df['reward'] = df['reward'].replace(0, pd.NA).ffill().bfill()

    df['entropy_norm'] = (df['entropy'] - df['entropy'].min()) / (df['entropy'].max() - df['entropy'].min())
    df['reward_norm'] = (df['reward'] - df['reward'].min()) / (df['reward'].max() - df['reward'].min())

    plt.plot(df[x_col], df['entropy_norm'], alpha=0.6, label="Entropy (norm)")

    df_sampled = df.iloc[::10]
    plt.plot(df_sampled[x_col], df_sampled['reward_norm'], alpha=0.6, label="Reward (norm)")

    plt.xlabel(x_col)
    plt.ylabel("Normalized Value (0-1)")
    plt.title("Entropy & Reward Normalized over Episodes")
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.grid()
    plt.savefig("plots/entropy_reward_normalized.png", dpi=150)
    plt.show()

def main():
    plot_reward_over_episode()
    plot_kd()
    plot_action_dist()
    plot_entropy_over_episode()
    plot_entropy_reward_normalized()

main()