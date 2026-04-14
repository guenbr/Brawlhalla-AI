import numpy as np

class PPOMemory:
    def __init__(self, use_cnn=False):
        self.use_cnn = use_cnn
        self.states = [] if use_cnn else None  # Only if CNN
        self.combined_data = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def store(self, state_or_data, combined_data_or_action, action_or_reward=None,
              reward=None, value=None, log_prob=None, done=None):
        """
        CNN mode: store(state, combined_data, action, reward, value, log_prob, done)
        MLP mode: store(combined_data, action, reward, value, log_prob, done)
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

    def clear(self):
        if self.use_cnn:
            self.states = []
        self.combined_data = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def get_batches(self):
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