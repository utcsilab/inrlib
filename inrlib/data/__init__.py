from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import numpy as np

class ABCDataset(ABC, Dataset):
    def __init__(self): 
        super().__init__()
        self.image: np.ndarray
        self.stage: bool
        self.input_shape = None
        self.x_data = None
        self.y_data = None
    
    @abstractmethod
    def change_stage(self): 
        pass