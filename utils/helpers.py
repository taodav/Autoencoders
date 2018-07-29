import torch
import torch.nn.functional as F
from argparse import ArgumentParser


def kl_divergence(p, q):
    p = F.softmax(p)
    q = F.softmax(q)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2

def get_args():
    parser = ArgumentParser(description='Image Autoencoder Experiment')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_hidden', type=int, default=8 * 3 * 3)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./saved')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args

