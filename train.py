from tqdm import tqdm
from utils import device
from utils.dataset import get_train_valid_loaders

class AutoencoderTrainer:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

        self.train_loader, self.valid_loader = get_train_valid_loaders(self.dataset)

    def train(self):
        raise NotImplementedError

    def train_step(self, image, label):
        image, label = image.to(device), label.to(device)
