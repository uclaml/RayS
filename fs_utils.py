import torch
import torch.nn as nn


class Attack_None(nn.Module):
    def __init__(self, basic_net, config):
        super(Attack_None, self).__init__()
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.basic_net = basic_net
        print(config)

    def forward(self, inputs):
        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()
        im_mean = torch.tensor([0.5, 0.5, 0.5]).cuda().view(1, inputs.shape[1], 1, 1).repeat(
            inputs.shape[0], 1, 1, 1)
        im_std = torch.tensor([0.5, 0.5, 0.5]).cuda().view(1, inputs.shape[1], 1, 1).repeat(
            inputs.shape[0], 1, 1, 1)
        processed = (inputs - im_mean) / im_std
        outputs, _ = self.basic_net(processed)
        return outputs


class Attack_Interp(nn.Module):
    def __init__(self, basic_net):
        super(Attack_Interp, self).__init__()
        self.basic_net = basic_net

    def forward(self, inputs):
        self.basic_net.eval()
        im_mean = torch.tensor([0.5, 0.5, 0.5]).cuda().view(1, inputs.shape[1], 1, 1).repeat(
            inputs.shape[0], 1, 1, 1)
        im_std = torch.tensor([0.5, 0.5, 0.5]).cuda().view(1, inputs.shape[1], 1, 1).repeat(
            inputs.shape[0], 1, 1, 1)
        processed = (inputs - im_mean) / im_std
        outputs = self.basic_net(processed)
        return outputs


class Attack_FS(nn.Module):
    def __init__(self, basic_net):
        super(Attack_FS, self).__init__()
        self.basic_net = basic_net

    def forward(self, inputs):
        im_mean = torch.tensor([0.5, 0.5, 0.5]).cuda().view(1, inputs.shape[1], 1, 1).repeat(
            inputs.shape[0], 1, 1, 1)
        im_std = torch.tensor([0.5, 0.5, 0.5]).cuda().view(1, inputs.shape[1], 1, 1).repeat(
            inputs.shape[0], 1, 1, 1)
        processed = (inputs - im_mean) / im_std
        return self.basic_net(processed)