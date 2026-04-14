import numpy as np


class PPOMemory:
    """
    Stores experience tuples collected during an episode for PPO training

    Supports two input modes:
        - CNN mode: stores screen frames alongside flat game state data
        - Non-CNN mode: stores flat game state data only
    """

    def __init__(self, use_cnn: bool = False) -> None:
        """
        Initialize memory buffers based on input mode

        Args:
            use_cnn (bool): whether to allocate a buffer for screen frames
        """
        self.use_cnn = use_cnn
        self.states = [] if use_cnn else None  # Only used in CNN mode
        self.combined_data = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def store(self, state_or_data: np.ndarray, combined_data_or_action: np.ndarray | int,
              action_or_reward: int | float | None = None, reward: float | None = None,
              value: float | None = None, log_prob: float | None = None,
              done: bool | None = None) -> None:
        """
        Store a single experience tuple in memory

        CNN mode: store(state, combined_data, action, reward, value, log_prob, done)
        Non-CNN mode: store(combined_data, action, reward, value, log_prob, done)

        Args:
            state_or_data: screen frame in CNN mode, or flat game state in non-CNN mode
            combined_data_or_action: flat game state in CNN mode, or action index in non-CNN mode
            action_or_reward: action index in CNN mode, or reward in non-CNN mode
            reward (float | None): reward received, CNN mode only
            value (float | None): critic value estimate, CNN mode only
            log_prob (float | None): log probability of the action taken, CNN mode only
            done (bool | None): whether the episode ended, CNN mode only
        """
        if self.use_cnn:
            self.states.append(state_or_data)
            self.combined_data.append(combined_data_or_action)
            self.actions.append(action_or_reward)
            self.rewards.append(reward)
            self.values.append(value)
            self.log_probs.append(log_prob)
            self.dones.append(done)
        else:
            self.combined_data.append(state_or_data)
            self.actions.append(combined_data_or_action)
            self.rewards.append(action_or_reward)
            self.values.append(reward)
            self.log_probs.append(value)
            self.dones.append(log_prob)

    def clear(self) -> None:
        """
        Clear all stored experiences after a PPO update
        """
        if self.use_cnn:
            self.states = []
        self.combined_data = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def get_batches(self) -> tuple:
        """
        Convert stored experience lists to numpy arrays for training

        Returns:
            tuple containing:
                - states (np.ndarray): screen frames, CNN mode only
                - combined_data (np.ndarray): flat game state vectors
                - actions (np.ndarray): action indices taken
                - rewards (np.ndarray): rewards received
                - values (np.ndarray): critic value estimates
                - log_probs (np.ndarray): log probabilities of actions taken
                - dones (np.ndarray): episode termination flags
        """
        if self.use_cnn:
            return (np.array(self.states),
                    np.array(self.combined_data),
                    np.array(self.actions),
                    np.array(self.rewards),
                    np.array(self.values),
                    np.array(self.log_probs),
                    np.array(self.dones))
        else:
            return (np.array(self.combined_data),
                    np.array(self.actions),
                    np.array(self.rewards),
                    np.array(self.values),
                    np.array(self.log_probs),
                    np.array(self.dones))
