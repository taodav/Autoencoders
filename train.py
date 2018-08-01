import torch
import os
# import time

from torch import nn
from tqdm import tqdm
from torchvision.utils import save_image
# from tensorboard_logger import configure, log_value
from utils import device
from utils.dataset import get_train_valid_loaders
from evaluate import calc_validation_loss
from utils.helpers import get_args, save_checkpoint, to_img, kl_divergence, load_checkpoint


class AutoencoderTrainer:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.args = get_args()
        # TO DO: refactor this?
        self.rho = torch.FloatTensor([0.05 for _ in range(self.model.hidden_size)]).unsqueeze(0).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        if self.args.resume_snapshot and os.path.isfile(self.args.resume_path):
            self.model, self.optimizer = load_checkpoint(self.args.resume_path, self.model, self.optimizer)

        # self.loss_func = nn.BCELoss()
        # self.loss_func.size_average = False
        self.loss_func = nn.MSELoss()

        self.train_loader, self.valid_loader = get_train_valid_loaders(self.dataset)
        # configure('./stats/%f' % (time.time()))

    def train(self):
        min_loss = float('inf')
        for ep in range(self.args.epochs):
            print("Training epoch %d" % (ep + 1))

            pbar = tqdm(enumerate(self.train_loader, 1), total=len(self.train_loader))
            running_loss_print = 0
            running_loss_save = 0
            for it, data in pbar:
                image, label = data
                loss, encoded, decoded = self.train_step(image, label)
                running_loss_print += loss
                running_loss_save += loss

                if it % self.args.log_every == 0:
                    pbar.set_description('it: %d, train_loss: %.6f' % (it, running_loss_print / self.args.log_every))
                    running_loss_print = 0

                if it % self.args.save_every == 0:
                    avg_running_loss = (running_loss_save / self.args.save_every)
                    is_best = min_loss > avg_running_loss
                    min_loss = avg_running_loss if is_best else min_loss
                    running_loss_save = 0

                    save_checkpoint({
                        'epoch': ep,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }, is_best)

            x = to_img(image.cpu().data)
            x_hat = to_img(decoded.cpu().data)
            save_image(x, './images/x_{}.png'.format(ep))
            save_image(x_hat, './images/x_{}_hat.png'.format(ep))

            valid_loss = calc_validation_loss(self.valid_loader, self.model, self.loss_func)

    def kl_loss(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        kl_loss /= mu.size(0) * self.model.hidden_size
        return kl_loss

    def train_step(self, image, label):
        image, label = image.to(device), label.to(device)
        encoded, decoded, mu, logvar = self.model(image)

        # image_flat = image.view(-1)
        # decoded_flat = decoded.view(-1)
        # loss = self.loss_func(decoded_flat, image_flat)

        loss = self.loss_func(decoded, image)

        if self.args.sparse:
            activations = encoded
            rho_hat = torch.sum(activations, dim=0, keepdim=True)
            sparsity_penalty = self.args.beta * kl_divergence(self.rho, rho_hat)
            loss += sparsity_penalty

        elif self.args.vae:
            loss += self.kl_loss(mu, logvar)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, encoded, decoded

