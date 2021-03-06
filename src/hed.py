# This file is adapted from ChipGAN.

import torch
import torch.nn as nn
import torchvision.models as models


class Hed(nn.Module):
    def __init__(self):
        super(Hed, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16 = self.vgg16.features
        for param in self.vgg16.parameters():
            param.requires_grad = False

        self.score_dsn1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn4 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn5 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)

        self.fuse = nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=0)

        self.upsample2 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(1, 1, kernel_size=6, stride=4, padding=1)
        self.upsample4 = nn.ConvTranspose2d(1, 1, kernel_size=12, stride=8, padding=2)
        self.upsample5 = nn.ConvTranspose2d(1, 1, kernel_size=24, stride=16, padding=4)

    def forward(self, x):
        cnt = 1
        res = []
        for l in self.vgg16:
            x = l(x)
            if cnt == 4:
                y = self.score_dsn1(x)
                res += [y]
            elif cnt == 9:
                y = self.score_dsn2(x)
                y = self.upsample2(y)
                res += [y]
            elif cnt == 16:
                y = self.score_dsn3(x)
                y = self.upsample3(y)
                res += [y]
            elif cnt == 23:
                y = self.score_dsn4(x)
                y = self.upsample4(y)
                res += [y]
            elif cnt == 30:
                y = self.score_dsn5(x)
                y = self.upsample5(y)
                res += [y]
            cnt += 1
        res = self.fuse(torch.cat(res, dim=1))
        return res

def no_sigmoid_cross_entropy(sig_logits, label):
    count_neg = torch.sum(1.-label)
    count_pos = torch.sum(label)

    beta = count_neg / (count_pos+count_neg)
    pos_weight = beta / (1-beta)

    cost = pos_weight * label * (-1) * torch.log(sig_logits) + (1-label)* (-1) * torch.log(1-sig_logits)
    cost = torch.mean(cost * (1-beta))

    return cost
