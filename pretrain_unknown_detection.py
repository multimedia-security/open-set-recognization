import argparse
import importlib
import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

import data_handler
from models.my_wide_resnet_embedding import Wide_ResNet
from models.utils import pprint, ensure_path
import random
import core_scripts.other_tools.list_tools as nii_list_tool
import core_scripts.data_io.default_data_io as nii_dset
import core_scripts.data_io.customize_collate_fn as nii_collate_fn


def train(epoch):
    print('\nEpoch:  %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_iterator):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if args.shmode == False and (batch_idx + 1) % 50 == 0:
            print(batch_idx, len(train_iterator), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    print('*******************test**********************')
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if args.shmode == False and (batch_idx + 1) % 50 == 0:
                print(batch_idx, len(test_iterator), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    acc = 100. * correct / total
    print('acc: {}'.format(acc))
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, osp.join(args.save_path, 'ckpt.pth'))
        best_acc = acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model_type', default='softmax', type=str, help='Recognition Method')
    parser.add_argument('--backbone', default='WideResnet', type=str, help='Backbone type.')
    parser.add_argument('--dataset', default='cifar10_relabel', type=str, help='dataset configuration')
    parser.add_argument('--gpu', default='0', type=str, help='use gpu')
    parser.add_argument('--upsampling', action='store_true', default=False,
                        help='Do not do upsampling.')
    parser.add_argument('--known_class', default=9, type=int, help='number of known class')
    parser.add_argument('--seed', default='9', type=int, help='random seed for dataset generation.')
    parser.add_argument('--shmode', action='store_true')
    parser.add_argument('--classes', type=int, default=9, help='最开始有多少类')

    args = parser.parse_args()
    args.cuda = True
    args.save_path = 'model_save_4'
    ensure_path(args.save_path, False)
    pprint(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing data..')

    prj_conf = importlib.import_module('config')

    # Loader used for training data

    # Load file list and create data loader
    trn_lst = nii_list_tool.read_list_from_text(prj_conf.proto_train_dir)
    trn_set = nii_dset.NIIDataSetLoader(
        'train',
        args,
        transforms,
        prj_conf.trn_eval_set_name,
        trn_lst,
        prj_conf.input_dirs,
        prj_conf.proto_train_dir,
        prj_conf.input_exts,
        prj_conf.input_dims,
        prj_conf.input_reso,
        prj_conf.input_norm,
        prj_conf.output_dirs,
        prj_conf.output_exts,
        prj_conf.output_dims,
        prj_conf.output_reso,
        prj_conf.output_norm,
        './',
        truncate_seq=prj_conf.truncate_seq,
        min_seq_len=prj_conf.minimum_len,
        save_mean_std=True,
        wav_samp_rate=prj_conf.wav_samp_rate)

    val_lst = nii_list_tool.read_list_from_text(prj_conf.proto_dev_dir)
    val_set = nii_dset.NIIDataSetLoader(
        'dev',
        args,
        transforms,
        prj_conf.val_eval_set_name,
        val_lst,
        prj_conf.val_input_dirs,
        prj_conf.proto_dev_dir,
        prj_conf.input_exts,
        prj_conf.input_dims,
        prj_conf.input_reso,
        prj_conf.input_norm,
        prj_conf.output_dirs,
        prj_conf.output_exts,
        prj_conf.output_dims,
        prj_conf.output_reso,
        prj_conf.output_norm,
        './',
        truncate_seq=prj_conf.truncate_seq,
        min_seq_len=prj_conf.minimum_len,
        save_mean_std=False,
        wav_samp_rate=prj_conf.wav_samp_rate)

    train_dataset_loader = data_handler.IncrementalLoader(trn_set,
                                                          args.classes, [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                                          cuda=args.cuda, oversampling=not args.upsampling,
                                                          )
    # Loader for test data.
    test_dataset_loader = data_handler.IncrementalLoader(val_set,
                                                         args.classes,
                                                         [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                                         cuda=args.cuda, oversampling=not args.upsampling,
                                                         )
    kwargs = {'num_workers': 8, 'pin_memory': False} if args.cuda else {}
    collate_fn = nii_collate_fn.customize_collate
    # Iterator to iterate over training data.
    train_iterator = torch.utils.data.DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=True,
                                                 **kwargs)
    # Iterator to iterate over test data
    test_iterator = torch.utils.data.DataLoader(test_dataset_loader, batch_size=args.batch_size, shuffle=True,
                                                **kwargs)

    # if args.dataset == 'cifar10_relabel':
    #     from data.cifar10_relabel import CIFAR10 as Dataset

    # trainset = Dataset('train', seed=args.seed)
    # knownlist, unknownlist = trainset.known_class_show()
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    # closeset = Dataset('testclose', seed=args.seed)
    # closerloader = torch.utils.data.DataLoader(closeset, batch_size=512, shuffle=True, num_workers=4)
    # openset = Dataset('testopen', seed=args.seed)
    # openloader = torch.utils.data.DataLoader(openset, batch_size=512, shuffle=True, num_workers=4)

    print('==> Building model..')
    if args.backbone == 'WideResnet':
        net = Wide_ResNet(28, 5, 0.3, args.known_class)
    net = net.to(device)
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # save_path1 = osp.join('results', 'D{}-M{}-B{}'.format(args.dataset, args.model_type, args.backbone, ))
    # save_path2 = 'LR{}-K{}-U{}-Seed{}'.format(str(args.lr), knownlist, unknownlist, str(args.seed))
    # args.save_path = osp.join(save_path1, save_path2)
    # ensure_path(save_path1, remove=False)
    # ensure_path(args.save_path, remove=False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.1)
    for epoch in range(start_epoch, start_epoch + 50):
        scheduler.step()
        train(epoch)
        if (epoch + 1) % 5 == 0:
            test(epoch)
            state = {'net': net.state_dict(), 'epoch': epoch, }
            torch.save(state, osp.join(args.save_path, 'Modelof_Epoch' + str(epoch) + '.pth'))
