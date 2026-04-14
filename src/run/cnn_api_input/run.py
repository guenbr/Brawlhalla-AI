from src.run.train import PPOTrainer


def run():
    trainer = PPOTrainer(
        gamma=0.97,
        lam=0.95,
        epsilon=0.10,
        epochs_per_update=1,
        entropy_coef=0.06,
        diversity_coef=0.0,
        episodes_per_update=4,
        num_episodes=500,
        learning_rate=0.00015,

        frame_skip=2,
        max_steps=30000,
        edge_threshold=0.15,
        edge_mask=False,
        starting_lives=5,
        use_cnn=True,
        # add input param size
        combined_data_size=8,

        log_every_n_steps=50,
        checkpoint_dir='checkpoints_9',
        log_dir='logs_9',

        entropy_warning=0.3,
        entropy_random=1.85,
        max_prob_spam=0.60,
        action_spam_pct=40,
    )
    trainer.train()


if __name__ == "__main__":
    run()