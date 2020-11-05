import argparse
import json
import numpy as np
import torch
import torchvision.models as models

from dataset import load_mnist_test_data, load_cifar10_test_data, load_imagenet_test_data
from general_torch_model import GeneralTorchModel

from arch import mnist_model
from arch import cifar_model

from RayS_Single import RayS




def main():
    parser = argparse.ArgumentParser(description='Hard Label Attacks')
    parser.add_argument('--dataset', default='imagenet', type=str,
                        help='Dataset')
    parser.add_argument('--targeted', default='0', type=str,
                        help='targeted or untargeted')
    parser.add_argument('--norm', default='linf', type=str,
                        help='Norm for attack, linf only')
    parser.add_argument('--num', default=10000, type=int,
                        help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--query', default=10000, type=int,
                        help='Maximum queries for the attack')
    parser.add_argument('--batch', default=1, type=int,
                        help='attack batch size.')
    parser.add_argument('--epsilon', default=0.05, type=float,
                        help='attack strength')
    parser.add_argument('--early', default='1', type=str,
                        help='early stopping (stop attack once the adversarial example is found)')
    args = parser.parse_args()

    targeted = True if args.targeted == '1' else False
    early_stopping = False if args.early == '0' else True
    order = 2 if args.norm == 'l2' else np.inf

    print(args)

    if args.dataset == 'mnist':
        model = mnist_model.MNIST().cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load('model/mnist_gpu.pt'))
        test_loader = load_mnist_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'cifar':
        model = cifar_model.CIFAR10().cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load('model/cifar10_gpu.pt'))
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'resnet':
        model = models.__dict__["resnet50"](pretrained=True).cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif args.dataset == 'inception':
        model = models.__dict__["inception_v3"](pretrained=True).cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    else:
        print("Invalid dataset")
        exit(1)

     
    attack = RayS(torch_model, order=order, epsilon=args.epsilon, early_stopping=early_stopping)
     
    stop_dists = []
    stop_queries = []
    asr = []
    np.random.seed(0)
    seeds = np.random.randint(10000, size=10000)
    count = 0
    for i, (xi, yi) in enumerate(test_loader):
        xi, yi = xi.cuda(), yi.cuda()

        if count == args.num:
            break

        if torch_model.predict_label(xi) != yi:
            continue

        np.random.seed(seeds[i])

        target = np.random.randint(torch_model.n_class) * torch.ones(yi.shape,
                                                                     dtype=torch.long).cuda() if targeted else None
        while target and torch.sum(target == yi) > 0:
            print('re-generate target label')
            target = np.random.randint(torch_model.n_class) * torch.ones(len(xi), dtype=torch.long).cuda()

        adv, queries, dist, succ = attack(xi, yi, target=target, seed=seeds[i],
                                          query_limit=args.query)
        # print(queries, dist, succ)
        if succ:
            stop_queries.append(queries)
            if dist.item() < np.inf:
                stop_dists.append(dist.item())
        elif early_stopping == False:
            if dist.item() < np.inf:
                stop_dists.append(dist.item())

        asr.append(succ.item())

        count += 1

        print("index: {:4d} avg dist: {:.4f} avg queries: {:.4f} asr: {:.4f} \n"
              .format(i,
                      np.mean(np.array(stop_dists)),
                      np.mean(np.array(stop_queries)),
                      np.mean(np.array(asr))
                      ))


    name = args.dataset + '_' + args.alg + '_' + args.norm + '_query' + str(args.query) + '_eps' + str(
        args.epsilon) + '_early' + args.early
    summary_txt = 'distortion: ' + str(np.mean(np.array(stop_dists))) + ' queries: ' + str(
        np.mean(np.array(stop_queries))) + ' succ rate: ' + str(np.mean(np.array(asr)))
    with open(name + '_summary' + '.txt', 'w') as f:
        json.dump(summary_txt, f)
 

if __name__ == "__main__":
    main()
