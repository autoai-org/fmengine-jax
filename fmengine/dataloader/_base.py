from itertools import cycle

class FMTrainerDataset(object):
    def get_stream(self):
        return cycle(self.get_sequence())

    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it