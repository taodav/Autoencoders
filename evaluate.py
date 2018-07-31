import torch
from utils import device


def calc_validation_loss(valid_loader, model, loss_func):
    with torch.no_grad():
        print("Calculating validation loss")
        valid_loss = 0
        for it, data in enumerate(valid_loader):
            image, label = data
            image, label = image.to(device), label.to(device)
            encoded, decoded, _, _ = model(image)
            loss = loss_func(decoded, image)
            valid_loss += loss

        print("validation loss: %.6f" % (valid_loss / it))

        return valid_loss


class AutoencoderEvaluator:
    def __init__(self, dataset, model):

        self.dataset = dataset
        self.model = model


    def print_decoded_with_image(self, image, decoded):
        raise NotImplementedError