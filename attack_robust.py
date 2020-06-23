import argparse
import numpy as np
import torch
import torchvision.models as models

from dataset import load_mnist_test_data, load_cifar10_test_data, load_imagenet_test_data
from general_torch_model import GeneralTorchModel
from mnist_model import MNIST
from cifar_model import CIFAR10
from wideresnet import WideResNet
from wideresnet1 import WideResNet1
from wideresnet2 import WideResNet2
from fs_utils import Attack_None, Attack_Interp, Attack_FS

from RaySBatch import RaySBatch

import json


def main():
    parser = argparse.ArgumentParser(description='Hard Label Attacks')
    parser.add_argument('--dataset', default='rob_cifar_trades', type=str,
                        help='Dataset')
    parser.add_argument('--targeted', default='0', type=str,
                        help='targeted or untargeted')
    parser.add_argument('--norm', default='linf', type=str,
                        help='Norm for attack, linf only')
    parser.add_argument('--num', default=50, type=int,
                        help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--query_limit', default=10000, type=int,
                        help='Maximum queries for the attack')
    parser.add_argument('--batch', default=10, type=int,
                        help='attack batch size.')
    parser.add_argument('--epsilon', default=0.05, type=float,
                        help='attack strength')
    parser.add_argument('--early', default='0', type=str,
                        help='early stopping (stop attack once the adversarial example is found)')
    args = parser.parse_args()

    targeted = True if args.targeted == '1' else False
    early_stopping = False if args.early == '0' else True
    ord = 2 if args.norm == 'l2' else np.inf

    print(args)

    if args.dataset == 'rob_cifar_trades':
        model = WideResNet().cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.module.load_state_dict(torch.load('model/rob_cifar_trades.pt'))
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'rob_cifar_adv':
        model = WideResNet().cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load('model/rob_cifar_madry.pt'))
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'rob_cifar_interp':
        model = WideResNet1(depth=28, num_classes=10, widen_factor=10).cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
        checkpoint = torch.load('model/rob_cifar_interp')
        model.load_state_dict(checkpoint['net'])
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=10, im_mean=[0.5, 0.5, 0.5],
                                        im_std=[0.5, 0.5, 0.5])
    elif args.dataset == 'rob_cifar_fs':
        config_natural = {'train': False}
        basic_net = WideResNet2(depth=28, num_classes=10, widen_factor=10).cuda()
        basic_net = basic_net.cuda()
        model = Attack_FS(basic_net, config_natural)
        model = torch.nn.DataParallel(model, device_ids=[0])
        checkpoint = torch.load('model/rob_cifar_fs')
        model.load_state_dict(checkpoint['net'])
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'rob_cifar_sense':
        model = WideResNet().cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load('model/SENSE_checkpoint300.dict')['state_dict'])
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=10, im_mean=None, im_std=None)
    else:
        print("Invalid dataset")
        exit(1)

    attack = RaySBatch(torch_model, ord=ord, epsilon=args.epsilon, early_stopping=early_stopping, args=args)

    stop_dists = []
    dists = []
    stop_queries = []
    asr = []
    adv = []

    np.random.seed(0)
    seeds = np.random.randint(10000, size=10000)
    count = 0
    for i, (xi, yi) in enumerate(test_loader):
        xi, yi = xi.cuda(), yi.cuda()

        if count >= args.num:
            break

        np.random.seed(seeds[i])

        target = np.random.randint(torch_model.n_class) * torch.ones(yi.shape,
                                                                     dtype=torch.long).cuda() if targeted else None
        while target and torch.sum(target == yi) > 0:
            print('re-generate target label')
            target = np.random.randint(torch_model.n_class) * torch.ones(len(xi), dtype=torch.long).cuda()

        adv_b, stop_queries_b, stop_dists_b, dists_b, asr_b = attack(xi, yi, target=target, seed=seeds[i],
                                                                     query_limit=args.query_limit)

        adv.append(adv_b)
        stop_queries.append(stop_queries_b)
        stop_dists.append(stop_dists_b)
        dists.append(dists_b)
        asr.append(asr_b)

        count += xi.shape[0]

        summary_txt = "index: {0} avg stop queries {1} avg stop dists {2} avg dists {3} asr {4} robust acc {5}\n" \
            .format(
            i,
            torch.stack(stop_queries).flatten().float().mean(),
            torch.stack(stop_dists).flatten().mean(),
            torch.stack(dists).flatten().mean(),
            torch.stack(asr).flatten().float().mean(),
            1 - torch.stack(asr).flatten().float().mean()
        )
        print(summary_txt)


    name = args.dataset + '_' + args.alg + '_' + args.norm + '_query' + str(args.query_limit) + '_eps' + str(
        args.epsilon) + '_early' + args.early + '_batch'
    with open(name + '_summary' + '.txt', 'w') as f:
        json.dump(summary_txt, f)


if __name__ == "__main__":
    main()
