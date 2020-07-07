import argparse
import json
import numpy as np
import torch


from dataset import load_mnist_test_data, load_cifar10_test_data, load_imagenet_test_data
from general_torch_model import GeneralTorchModel
from general_tf_model import GeneralTFModel

from arch import fs_utils
from arch import wideresnet
from arch import wideresnet_fs
from arch import wideresnet_interp
from arch import wideresnet_he
from arch import wideresnet_rst
from arch import wideresnet_overfitting
from arch import madry_wrn
from arch import preact_resnet
from arch import wideresnet_compact

from RayS import RayS

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

np.random.seed(1234)


def main():
    parser = argparse.ArgumentParser(description='RayS Attacks')
    parser.add_argument('--dataset', default='rob_cifar_trades', type=str,
                        help='robust model / dataset')
    parser.add_argument('--targeted', default='0', type=str,
                        help='targeted or untargeted')
    parser.add_argument('--norm', default='linf', type=str,
                        help='Norm for attack, linf only')
    parser.add_argument('--num', default=10000, type=int,
                        help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--query', default=10000, type=int,
                        help='Maximum queries for the attack')
    parser.add_argument('--batch', default=10, type=int,
                        help='attack batch size.')
    parser.add_argument('--epsilon', default=0.05, type=float,
                        help='attack strength')
    args = parser.parse_args()

    targeted = True if args.targeted == '1' else False
    order = 2 if args.norm == 'l2' else np.inf

    print(args)
    summary_all = ''

    if args.dataset == 'rob_cifar_trades':
        model = wideresnet.WideResNet().cuda()
        model = torch.nn.DataParallel(model)
        model.module.load_state_dict(torch.load('model/rob_cifar_trades.pt'))
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(
            model, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'rob_cifar_adv':
        model = wideresnet.WideResNet().cuda()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('model/rob_cifar_madry.pt'))
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(
            model, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'rob_cifar_madry':
        import tensorflow as tf
        model = madry_wrn.Model(mode='eval')
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, tf.train.latest_checkpoint('model/madry'))
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTFModel(
            model.pre_softmax, model.x_input, sess, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'rob_cifar_interp':
        model = wideresnet_interp.WideResNet(
            depth=28, num_classes=10, widen_factor=10).cuda()
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load('model/rob_cifar_interp')
        model.load_state_dict(checkpoint['net'])
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=10, im_mean=[0.5, 0.5, 0.5],
                                        im_std=[0.5, 0.5, 0.5])
    elif args.dataset == 'rob_cifar_fs':
        basic_net = wideresnet_fs.WideResNet(
            depth=28, num_classes=10, widen_factor=10).cuda()
        basic_net = basic_net.cuda()
        model = fs_utils.Model_FS(basic_net)
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load('model/rob_cifar_fs')
        model.load_state_dict(checkpoint['net'])
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=10, im_mean=[0.5, 0.5, 0.5],
                                        im_std=[0.5, 0.5, 0.5])
    elif args.dataset == 'rob_cifar_sense':
        model = wideresnet.WideResNet().cuda()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(
            'model/SENSE_checkpoint300.dict')['state_dict'])
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(
            model, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'rob_cifar_rst':
        model = wideresnet_rst.WideResNet_RST()
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load(
            'model/rst_adv.pt.ckpt')['state_dict'])
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(
            model, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'rob_cifar_mart':
        model = wideresnet_rst.WideResNet_RST().cuda()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(
            'model/mart_unlabel.pt')['state_dict'])
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(
            model, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'rob_cifar_uat':
        import tensorflow_hub as hub
        import tensorflow as tf
        UAT_HUB_URL = ('./model/uat_model')
        model = hub.Module(UAT_HUB_URL)
        my_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        my_logits = model(dict(x=my_input, decay_rate=0.1, prefix='default'))
        sess = tf.Session()
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTFModel(
            my_logits, my_input, sess, n_class=10, im_mean=[125.3/255, 123.0/255, 113.9/255], im_std=[63.0/255, 62.1/255, 66.7/255])
    elif args.dataset == 'rob_cifar_overfitting':
        model = wideresnet_overfitting.WideResNet(depth=34, num_classes=10, widen_factor=20).cuda()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('model/rob_cifar_overfitting.pth'))
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(
            model, n_class=10, im_mean=[0.4914, 0.4822, 0.4465], im_std=[0.2471, 0.2435, 0.2616])
    elif args.dataset == 'rob_cifar_pretrain':
        model = wideresnet_overfitting.WideResNet(depth=28, num_classes=10, widen_factor=10).cuda()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('model/rob_cifar_pretrain.pt'))
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(
            model, n_class=10, im_mean=[0.5, 0.5, 0.5], im_std=[0.5, 0.5, 0.5])
    elif args.dataset == 'rob_cifar_fast':
        model = preact_resnet.PreActResNet18().cuda()
        model.load_state_dict(torch.load('model/rob_cifar_fast_epoch30.pth'))
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(
            model, n_class=10, im_mean=[0.4914, 0.4822, 0.4465], im_std=[0.2471, 0.2435, 0.2616])
    elif args.dataset == 'rob_cifar_compact':
        model = torch.nn.DataParallel(wideresnet_compact.wrn_28_10())
        ckpt = torch.load('model/rob_cifar_compact.pth.tar', map_location="cpu")["state_dict"]
        model.load_state_dict(ckpt)
        model.cuda()
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(
            model, n_class=10, im_mean=None, im_std=None)
    if args.dataset == 'rob_cifar_mma':
        from advertorch_examples.models import get_cifar10_wrn28_widen_factor
        model = get_cifar10_wrn28_widen_factor(4).cuda()
        model = torch.nn.DataParallel(model)
        model.module.load_state_dict(torch.load('model/rob_cifar_mma.pt')['model'])
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(
            model, n_class=10, im_mean=None, im_std=None)
    if args.dataset == 'rob_cifar_he':
        model = wideresnet_he.WideResNet(normalize = True).cuda()
        model = torch.nn.DataParallel(model)
        model.module.load_state_dict(torch.load('model/rob_cifar_pgdHE.pt'))
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(
            model, n_class=10, im_mean=None, im_std=None)
    else:
        print("Invalid dataset")
        exit(1)

    attack = RayS(torch_model, epsilon=args.epsilon, order=order)

    adbd = []
    queries = []
    succ = []

    count = 0
    for i, (data, label) in enumerate(test_loader):
        data, label = data.cuda(), label.cuda()

        if count >= args.num:
            break

        if targeted:
            target = np.random.randint(torch_model.n_class) * torch.ones(
                label.shape, dtype=torch.long).cuda() if targeted else None
            while target and torch.sum(target == label) > 0:
                print('re-generate target label')
                target = np.random.randint(
                    torch_model.n_class) * torch.ones(len(data), dtype=torch.long).cuda()
        else:
            target = None

        _, queries_b, adbd_b, succ_b = attack(
            data, label, target=target, query_limit=args.query)

        queries.append(queries_b)
        adbd.append(adbd_b)
        succ.append(succ_b)

        count += data.shape[0]

        summary_batch = "Batch: {:4d} Avg Queries (when found adversarial examples): {:.4f} ADBD: {:.4f} Robust Acc: {:.4f}\n" \
            .format(
                i + 1,
                torch.stack(queries).flatten().float().mean(),
                torch.stack(adbd).flatten().mean(),
                1 - torch.stack(succ).flatten().float().mean()
            )
        print(summary_batch)
        summary_all += summary_batch

    name = args.dataset + '_query_' + str(args.query) + '_batch'
    with open(name + '_summary' + '.txt', 'w') as fileopen:
        json.dump(summary_all, fileopen)


if __name__ == "__main__":
    main()
