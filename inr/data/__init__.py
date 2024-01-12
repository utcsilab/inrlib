from torch.utils.data import Dataset
from abc import ABC

class ABCDataset(ABC, Dataset):
    def __init__(self): 
        super().__init__()
        self.image
        self.input_shape = None
        self.x_data = None
        self.y_data = None