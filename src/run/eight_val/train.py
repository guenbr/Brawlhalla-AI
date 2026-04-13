from src.run.train import PPOTrainer


def run():
    trainer = PPOTrainer(
        gamma=0.995,
        lam=0.95,
        epsilon=0.10,
        epochs_per_update=4,
        entropy_coef=0.01,
        diversity_coef=0.03,
        episodes_per_update=8,
        num_episodes=500,
        learning_rate=0.00015,

        frame_skip=2,
        max_steps=30000,
        edge_threshold=0.15,
        edge_mask=False,
        starting_lives=10,
        use_cnn=False,
        combined_data_size=8,

        log_every_n_steps=50,
        checkpoint_dir='run/eight_val/checkpoints_2',
        log_dir='run/eight_val/logs_2',

        entropy_warning=0.5,
        entropy_random=1.85,
        max_prob_spam=0.60,
        action_spam_pct=40,
    )
    trainer.train()


if __name__ == "__main__":
    run()
