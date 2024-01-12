from abc import ABC, abstractmethod
import lightning.pytorch as pl


class ABCModel(ABC, pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.val_outputs = []
		self.scores = None
  
	def on_validation_start(self):
		self.val_outputs = []
		self.scores = None
		
	@abstractmethod
	def reconstruct(self):
		pass
	
	@abstractmethod
	def compute_metrics(self):
		pass