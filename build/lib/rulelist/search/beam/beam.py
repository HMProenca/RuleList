import numpy as np


class Beam():
    def __init__(self, beam_width):
        self.beam_width = beam_width
        self.patterns = [[] for w in range(beam_width)]
        self.array_score = np.full(beam_width, np.NINF)
        self.array_support = np.full(beam_width, np.inf)
        self.min_support_beam = np.inf
        self.set_patterns = [set() for w in range(beam_width)]
        self.min_score = np.NINF
        self.min_index = 0

    def replace(self, new_pattern, new_score, usage):
        self.patterns[self.min_index] = new_pattern
        self.set_patterns[self.min_index] = set([item.description for item in new_pattern])
        self.array_score[self.min_index] = new_score
        self.array_support[self.min_index] = usage
        self.min_index = self.array_score.argmin()
        self.min_score = self.array_score[self.min_index]
        if usage < self.min_support_beam:
            self.min_support_beam = usage
        return self

    def clean(self):
        self.patterns = [[] for w in range(self.beam_width)]
        self.set_patterns = [set() for pat in self.patterns]
        self.array_score = np.full(self.beam_width, np.NINF)
        self.array_support = np.full(self.beam_width, np.inf)
        self.min_score = np.NINF
        self.min_index = 0
        return self