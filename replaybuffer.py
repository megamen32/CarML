from collections import deque
import random
import numpy as np
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
import heapq

class PriorityReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha=0.5

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        max_priority = max((self.priorities)) if len(self.buffer) > 0 else 1.0

        self.buffer.append(experience)
        if isinstance(max_priority, (int, float, np.number)):
            self.priorities.append(float(max_priority))
        else:
            print("Warning: Trying to append a non-scalar value to priorities.")

    def sample(self, batch_size):
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        return samples, indices, probs[indices]

    def update_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = e + offset

    def total_priority(self):
        return sum(self.priorities)

    def __len__(self):
        return len(self.buffer)


