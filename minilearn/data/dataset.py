import numpy as np

class DATASET :
    def __inti__(self,*data):
        self.array = tuple(np.asarray(arr) for arr in data)

    def __len__(self):
        return len(self.array[0])
    def __get__item(self,idx):
        items = tuple(arr[idx] for arr in self.array)
        if len(items) == 1:return items[0] 
        else: return items