import torch
import os
from torch import nn
from tqdm import tqdm
from torchvision.utils import save_image
from utils import device
from utils.dataset import get_train_valid_loaders
from utils.helpers import get_args, save_checkpoint, to_img, kl_divergence

class AutoencoderTrainer:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.args = get_args()
        # TO DO: refactor this?
        self.rho = torch.FloatTensor([0.05 for _ in range(self.model.hidden_size)]).unsqueeze(0).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        if self.args.resume_snapshot and os.path.isfile(self.args.resume_path):
            print("Loading checkpoint")
            checkpoint = torch.load(self.args.resume_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint")

        self.loss_func = nn.MSELoss()
        self.train_loader, self.valid_loader = get_train_valid_loaders(self.dataset)

    def train(self):
        min_loss = float('inf')
        for ep in range(self.args.epochs):
            print("Training epoch %d" % (ep + 1))

            pbar = tqdm(enumerate(self.dataset, 1), total=len(self.dataset))
            epoch_loss = 0
            for it, data in pbar:
                image, label = data
                loss, encoded, decoded = self.train_step(image, label)
                epoch_loss += loss

                if it % self.args.log_every == 0:
                    pbar.set_description('it: %d, train_loss: %.6f' % (it, epoch_loss / it))

                if it % self.args.save_every == 0:
                    is_best = min_loss > epoch_loss
                    min_loss = epoch_loss if is_best else min_loss
                    save_checkpoint({
                        'epoch': ep,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }, is_best)
                if it % 10000 == 0:
                    x = to_img(image.cpu().data)
                    x_hat = to_img(decoded.cpu().data)
                    save_image(x, './images/x_{}.png'.format(it))
                    save_image(x_hat, './images/x_hat_{}.png'.format(it))

            with torch.no_grad():
                print("Calculating validation loss")
                valid_loss = 0
                for it, data in enumerate(self.valid_loader):
                    vs_loss, encoded, decoded = self.train_step(*data)
                    valid_loss += vs_loss

                print("validation loss: %.6f" % (valid_loss / it))

    def train_step(self, image, label):
        image, label = image.to(device), label.to(device)
        image = image.view(-1, 28 * 28)
        encoded, decoded = self.model(image)

        loss = self.loss_func(decoded, image)

        if self.args.sparse:
            activations = encoded
            rho_hat = torch.sum(activations, dim=0, keepdim=True)
            sparsity_penalty = self.args.beta * kl_divergence(self.rho, rho_hat)
            loss += sparsity_penalty

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, encoded, decoded