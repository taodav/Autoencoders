
from utils.dataset import get_train_valid_loaders

class AutoencoderTrainer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_loader, self.valid_loader = get_train_valid_loaders(self.dataset)

    def train(self):
        raise NotImplementedError
