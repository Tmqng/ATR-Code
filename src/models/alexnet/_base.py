import torch


class ModelTrainer(object):
    def __init__(self, model, **params):
        self.model = model
        self.model_str = model.model_str

        self.loss_fn = params.get("loss_fn")
        self.optimizer = params.get("optimizer")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

