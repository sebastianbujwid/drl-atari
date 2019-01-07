import numpy as np


class ExperienceReplay:

    def __init__(self, max_size):
        self.max_size = max_size
        self._experiences = []

    def add(self, state, action, reward, done, next_state):
        self._experiences.append((state, action, reward, done, next_state))
        num_elements_to_remove = len(self._experiences) - self.max_size
        if num_elements_to_remove > 0:
            del self._experiences[:num_elements_to_remove]

    def sample(self, num_elements):
        indices = np.random.choice(self.size(), size=num_elements, replace=False)
        samples = [self._experiences[i] for i in indices]
        return samples

    def size(self):
        return len(self._experiences)