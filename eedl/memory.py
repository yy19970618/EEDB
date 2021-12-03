from collections import deque
import random


class Memory:
    # memory_store = deque()
    memory_size = 120000

    def __init__(self):
        self.replay_memory_store = deque()

    def save_store(self, query, card):
        self.replay_memory_store.append((query, card))
        if len(self.replay_memory_store) > self.memory_size:
            self.replay_memory_store.popleft()

    def sample(self, batch):
        minibatch = random.sample(self.replay_memory_store, batch)
        return minibatch

    def clear(self):
        self.replay_memory_store.clear()

    def size(self):
        return len(self.replay_memory_store)
