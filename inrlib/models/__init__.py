from abc import ABC, abstractmethod
import lightning.pytorch as pl


class ABCModel(ABC, pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.outputs = []
		self.scores = None
  
	def on_validation_start(self):
		self.outputs = []
		self.scores = None
		super().on_validation_start()
	
	def on_test_start(self):
		self.outputs = []
		self.scores = None
		super().on_test_start()
  
	def on_predict_start(self) -> None:
		self.outputs = []
		self.scores = None
		super().on_predict_start()
		
	@abstractmethod
	def reconstruct(self):
		pass
	
	@abstractmethod
	def compute_metrics(self):
		pass