from src.run.train import PPOTrainer


def run():
    trainer = PPOTrainer(
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
        combined_data_size=14,

        # Logging settings
        log_every_n_steps=50,
        checkpoint_dir='checkpoints_14',
        log_dir='logs_14',

        # Warning thresholds
        entropy_warning=0.5,
        entropy_random=1.85,
        max_prob_spam=0.60,
        action_spam_pct=40,
    )
    trainer.train()


if __name__ == "__main__":
    run()