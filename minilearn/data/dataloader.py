import numpy as np
from .dataset import DATASET


class Dataloader:
    def __init__(self,dataset,bach_size=32,shuffle=True,drop_last=False):
        if isinstance(dataset,DATASET):
            self.dataset = dataset
        elif isinstance(dataset,np.ndarray):
            self.dataset = DATASET(dataset)
        elif isinstance(dataset,(tuple,list)):
            self.dataset = DATASET(*dataset)

        self.batch_size = int(bach_size)
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
            n = len(self.dataset)
            if self.drop_last:
                 return n // self.batch_size
            return (n + self.batch_size - 1) // self.batch_size
        