import numpy as np


class PPOMemory:
    def __init__(self):
        self.combined_data = []
        self.actions       = []
        self.rewards       = []
        self.values        = []
        self.log_probs     = []
        self.dones         = []

    def store(self, combined_data, action, reward, value, log_prob, done):
        self.combined_data.append(combined_data)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.combined_data = []; self.actions = []
        self.rewards = []; self.values = []; self.log_probs = []; self.dones = []

    def get_batches(self):
        return (np.array(self.combined_data),
                np.array(self.actions),
                np.array(self.rewards),
                np.array(self.values),
                np.array(self.log_probs),
                np.array(self.dones))
