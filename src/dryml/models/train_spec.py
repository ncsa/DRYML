import os
import pickle
from typing import List

class TrainStateException(Exception):
    pass

class TrainSpec(object):
    def __init__(self, level_steps: List[int] = None, global_steps=0):
        if level_steps is None:
            self.level_steps = [0]
        else:
            self.level_steps = level_steps
        self.global_steps = global_steps
        self.cur_level = 0

    def is_state_current(self):
        """
        Check spec state for preparedness to modify
        """
        return self.cur_level == len(self.level_steps)-1

    def descend(self):
        if self.is_state_current():
            self.level_steps.append(0)
            self.global_steps += 1
        self.cur_level += 1

    def elevate(self):
        if not self.is_state_current():
            raise TrainStateException()
        self.level_steps.pop()
        self.cur_level -= 1

    def advance(self):
        if not self.is_state_current():
            raise TrainStateException()
        self.global_steps += 1
        self.level_steps[self.cur_level] += 1

    def level_step(self):
        return self.level_steps[self.cur_level]

    def global_step(self):
        return self.global_steps

    def __str__(self):
        return f"{self.level_steps}, {self.global_steps} ({self.cur_level})"

    def __repr__(self):
        return f"{self.level_steps}, {self.global_steps} ({self.cur_level})"

    def save(self, path):
        with open(path, 'wb') as f:
            f.write(pickle.dumps({'level_steps': self.level_steps, 'global_steps': self.global_steps}))

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            def_dict = pickle.loads(f.read())
        return TrainSpec(level_steps=def_dict['level_steps'], global_steps=def_dict['global_steps'])
