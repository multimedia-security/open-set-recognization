import argparse
import importlib
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from models.wide_proto_resnet import Wide_ResNet
from models.proto_type import DceLoss
import random
import core_scripts.other_tools.list_tools as nii_list_tool
import core_scripts.data_io.default_data_io as nii_dset
import core_scripts.data_io.customize_collate_fn as nii_collate_fn
import data_handler
from mylogger import mylogger


def progress_bar(s, total, str):
    c = (s + 1) / total * 100
    s = s / total * 50
    total = 50
    start = '*' * int(s + 1)
    end = '.' * int(total - s)
    print('\r{:.2f}%[{}->{}]{}'.format(c, start, end, str), end='')


def evaluate(target, pred, nb_per_class, correct_per_class):
    for i in range(len(pred)):
        nb_per_class[target[i].detach().cpu()] += 1
        if target[i] == pred[i]:
            correct_per_class[target[i].detach().cpu()] += 1
    # correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    for j in range(correct_per_class.shape[0]):
        mylogger.info('class: {} correct rate = 100 * {} / {} = {}'.format(j, correct_per_class[j], nb_per_class[j],
                                                                           100 * correct_per_class[j] / nb_per_class[
                                                                               j]))

    # return 100. * correct / len(loader.dataset)
    return nb_per_class, correct_per_class


def traindummy(epoch, net, dce):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params': net.parameters()}, {'params': dce.parameters()}], lr=args.lr * 0.01, momentum=0.9,
                          weight_decay=args.decay)

    mylogger.debug('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    alpha = args.alpha
    for batch_idx, (inputs, targets) in enumerate(train_iterator):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        totallenth = len(inputs)
        halflenth = int(len(inputs) / 2)
        beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()

        prehalfinputs = inputs[:halflenth]
        prehalflabels = targets[:halflenth]
        laterhalfinputs = inputs[halflenth:]
        laterhalflabels = targets[halflenth:]

        # 随机组合前半个size的中间特征
        index = torch.randperm(prehalfinputs.size(0)).cuda()
        pre2embeddings = pre2block(net, prehalfinputs)
        mixed_embeddings = beta * pre2embeddings + (1 - beta) * pre2embeddings[index]  # 产生虚拟数据点

        dummylogit, o_dummylogit, out = dummypredict(net, dce, laterhalfinputs)  # 输出batch_size,1的向量
        outputs1, outputs2 = net(laterhalfinputs)  # 正常输出batch_size,6的向量
        centers, lateroutputs = dce(outputs2)
        # print(centers)
        latterhalfoutput = torch.cat((lateroutputs, dummylogit), 1)  # 将上面两个向量组合为batch_size，7的向量
        o_latterhalfoutput = torch.cat((outputs1, o_dummylogit), 1)
        # re_output = torch.cat((outputs2, out), 1)
        # 将前半个batch_size的数据(经过混合)完成网络输出，并组合为batch_size，7的向量
        l2b1, o_l2b1, mix_feat = latter2blockclf1(net, dce, mixed_embeddings)
        l2b2, o_l2b2 = latter2blockclf2(net, dce, mixed_embeddings)
        prehalfoutput = torch.cat((l2b1, l2b2), 1)
        o_prehalfoutput = torch.cat((o_l2b1, o_l2b2), 1)

        maxdummy, _ = torch.max(dummylogit.clone(), dim=1)  # 对batch_size，1的向量进行softmax
        o_maxdummy, _ = torch.max(o_dummylogit.clone(), dim=1)
        maxdummy = maxdummy.view(-1, 1)
        o_maxdummy = maxdummy.view(-1, 1)
        dummpyoutputs = torch.cat((lateroutputs.clone(), maxdummy), dim=1)  # 将batch_size,6的向量和batch_size,1的向量组合
        o_dummpyoutputs = torch.cat((outputs1.clone(), o_maxdummy), dim=1)
        # 将原来的标签置信度减小
        for i in range(len(dummpyoutputs)):
            nowlabel = laterhalflabels[i]
            dummpyoutputs[i][nowlabel] = -1e9
            o_dummpyoutputs[i][nowlabel] = -1e9
        dummytargets = torch.ones_like(laterhalflabels) * args.known_class

        outputs = torch.cat((prehalfoutput, latterhalfoutput), 0)
        # 插入的虚拟类损失
        loss1 = 1 * criterion(prehalfoutput,
                              (torch.ones_like(prehalflabels) * args.known_class).long().cuda()) + criterion(
            o_prehalfoutput, (torch.ones_like(prehalflabels) * args.known_class).long().cuda())
        # 真实类的损失
        loss2 = 1 * criterion(latterhalfoutput, laterhalflabels) + criterion(o_latterhalfoutput, laterhalflabels)
        # 第二置信度损失
        loss3 = 1 * criterion(dummpyoutputs, dummytargets) + criterion(o_dummpyoutputs, dummytargets)
        loss4 = dce.regularization(outputs2, laterhalflabels) * 0.1
        loss5 = F.mse_loss(mix_feat, torch.t(dce.dummy_centers)) * 0.1
        loss = 0.1 * loss1 + args.lamda1 * loss2 + args.lamda2 * loss3 + loss4 + loss5

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # if args.shmode == False and (batch_idx + 1) % 20 == 0:
        progress_bar(batch_idx, len(train_iterator),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d) L1 %.3f, L2 %.3f, L3 %.3f, L4 %.3f, L5 %.3f' % (
                         train_loss / (batch_idx + 1), 100. * correct / total, correct, total, loss1.item(),
                         loss2.item(), loss3.item(), loss4.item(), loss5.item()))


def valdummy(epoch, net, dce, mainepoch):
    net.eval()
    # mylogger.info('验证')
    CONF_AUC = False
    CONF_DeltaP = True
    auclist1 = []
    auclist2 = []
    linspace = [0]
    open_preds = []
    close_preds = []
    all_targets = []
    all_preds = []
    close_nb = np.zeros(20)
    open_nb = np.zeros(20)
    close_cor = np.zeros(20)
    open_cor = np.zeros(20)
    closelogits = torch.zeros((len(test_iterator.dataset), args.known_class + 1)).cuda()
    openlogits = torch.zeros((len(novel_iterator.dataset), args.known_class + 1)).cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            all_targets.extend(targets)
            batchnum = len(targets)
            _, outputs2 = net(inputs)
            centers, logits = dce(outputs2)
            dummylogit, _, _ = dummypredict(net, dce, inputs)
            maxdummylogit, _ = torch.max(dummylogit, 1)
            maxdummylogit = maxdummylogit.view(-1, 1)
            totallogits = torch.cat((logits, maxdummylogit), dim=1)
            closelogits[batch_idx * batchnum:batch_idx * batchnum + batchnum, :] = totallogits
        for batch_idx, (inputs, targets) in enumerate(novel_iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            # all_targets.extend(targets)
            batchnum = len(targets)
            _, outputs2 = net(inputs)
            centers, logits = dce(outputs2)
            dummylogit, _, _ = dummypredict(net, dce, inputs)
            maxdummylogit, _ = torch.max(dummylogit, 1)
            maxdummylogit = maxdummylogit.view(-1, 1)
            totallogits = torch.cat((logits, maxdummylogit), dim=1)
            openlogits[batch_idx * batchnum:batch_idx * batchnum + batchnum, :] = totallogits
    Logitsbatchsize = 200
    maxauc = 0
    maxaucbias = 0
    for biasitem in linspace:
        if CONF_AUC:
            for temperature in [1024.0]:
                closeconf = []
                openconf = []
                closeiter = int(len(closelogits) / Logitsbatchsize)
                openiter = int(len(openlogits) / Logitsbatchsize)
                for batch_idx in range(closeiter):
                    logitbatch = closelogits[batch_idx * Logitsbatchsize:batch_idx * Logitsbatchsize + Logitsbatchsize,
                                 :]
                    logitbatch[:, -1] = logitbatch[:, -1] + biasitem
                    embeddings = nn.functional.softmax(logitbatch / temperature, dim=1)
                    conf = embeddings[:, -1]
                    closeconf.append(conf.cpu().numpy())
                closeconf = np.reshape(np.array(closeconf), (-1))
                closelabel = np.ones_like(closeconf)
                for batch_idx in range(openiter):
                    logitbatch = openlogits[batch_idx * Logitsbatchsize:batch_idx * Logitsbatchsize + Logitsbatchsize,
                                 :]
                    logitbatch[:, -1] = logitbatch[:, -1] + biasitem
                    embeddings = nn.functional.softmax(logitbatch / temperature, dim=1)
                    conf = embeddings[:, -1]
                    openconf.append(conf.cpu().numpy())
                openconf = np.reshape(np.array(openconf), (-1))
                openlabel = np.zeros_like(openconf)
                totalbinary = np.hstack([closelabel, openlabel])
                totalconf = np.hstack([closeconf, openconf])
                auc1 = roc_auc_score(1 - totalbinary, totalconf)
                auc2 = roc_auc_score(totalbinary, totalconf)
                print('Temperature:', temperature, 'bias', biasitem, 'AUC_by_confidence', auc2)
                auclist1.append(np.max([auc1, auc2]))
        if CONF_DeltaP:
            for temperature in [1024.0]:
                closeconf = []
                openconf = []
                closeiter = int(len(closelogits) / Logitsbatchsize)
                openiter = int(len(openlogits) / Logitsbatchsize)
                for batch_idx in range(closeiter):
                    logitbatch = closelogits[batch_idx * Logitsbatchsize:batch_idx * Logitsbatchsize + Logitsbatchsize,
                                 :]
                    logitbatch[:, -1] = logitbatch[:, -1] + biasitem
                    embeddings = nn.functional.softmax(logitbatch / temperature, dim=1)
                    dummyconf = embeddings[:, -1].view(-1, 1)
                    maxknownconf, pred = torch.max(embeddings[:, :-1], dim=1)
                    all_preds.extend(pred)
                    maxknownconf = maxknownconf.view(-1, 1)
                    conf = dummyconf - maxknownconf
                    for c in conf:
                        if c < 0:
                            close_preds.append(1)
                        else:
                            close_preds.append(0)
                    closeconf.append(conf.cpu().numpy())
                close_nb, close_cor = evaluate(all_targets, all_preds, close_nb, close_cor)
                closeconf = np.reshape(np.array(closeconf), (-1))
                closelabel = np.ones_like(closeconf)
                for batch_idx in range(openiter):
                    logitbatch = openlogits[batch_idx * Logitsbatchsize:batch_idx * Logitsbatchsize + Logitsbatchsize,
                                 :]
                    logitbatch[:, -1] = logitbatch[:, -1] + biasitem
                    embeddings = nn.functional.softmax(logitbatch / temperature, dim=1)
                    dummyconf = embeddings[:, -1].view(-1, 1)
                    maxknownconf, _ = torch.max(embeddings[:, :-1], dim=1)
                    maxknownconf = maxknownconf.view(-1, 1)
                    conf = dummyconf - maxknownconf
                    for c in conf:
                        if c > 0:
                            open_preds.append(0)
                        else:
                            open_preds.append(1)
                    openconf.append(conf.cpu().numpy())
                openconf = np.reshape(np.array(openconf), (-1))
                openlabel = np.zeros_like(openconf)
                totalbinary = np.hstack([closelabel, openlabel])
                totalconf = np.hstack([closeconf, openconf])
                opencompare = np.equal(open_preds, openlabel).sum()
                closecompare = np.equal(close_preds, closelabel).sum()
                mylogger.info(
                    '检测close正确率：{}/{}={}'.format(closecompare, len(close_preds), closecompare / len(close_preds)))
                mylogger.info(
                    '检测open正确率：{}/{}={}'.format(opencompare, len(open_preds), opencompare / len(open_preds)))
                auc1 = roc_auc_score(1 - totalbinary, totalconf)
                auc2 = roc_auc_score(totalbinary, totalconf)
                mylogger.info(
                    'Temperature: {} bias: {} AUC_by_Delta_confidence: {}'.format(temperature, biasitem, auc1))
                auclist1.append(np.max([auc1, auc2]))
    return np.max(np.array(auclist1))


def finetune_proser(epoch=59):
    print('Now processing epoch', epoch)
    net = getmodel(args)
    print('==> Resuming from checkpoint..')
    # assert os.path.isdir(model_path), 'Error: no checkpoint directory found!'
    modelname = 'Modelof_Epoch' + str(epoch) + '.pth'
    checkpoint = torch.load('model_save_proto2/Modelof_Epoch19.pth')
    net.load_state_dict(checkpoint['net'])

    net.clf2 = nn.Linear(3840, args.dummynumber)
    net = net.cuda()
    dce = DceLoss(args.known_class, args.feat_dim)
    dce.load_state_dict(checkpoint['dce'])
    dce.dummy_centers = torch.nn.Parameter(torch.randn(args.feat_dim, args.dummynumber).cuda(), requires_grad=True)

    FineTune_MAX_EPOCH = 20
    wholebestacc = 0
    for finetune_epoch in range(FineTune_MAX_EPOCH):
        start_time = time.time()
        traindummy(finetune_epoch, net, dce)
        if (finetune_epoch + 1) % 5 == 0:
            finetuneacc = valdummy(finetune_epoch, net, dce, epoch)
            # print(finetuneacc)
            state = {
                'net': net.state_dict(),
                'dce': dce.state_dict(),
                'epoch': epoch,
            }
            model_path = os.path.join(model_dir, '{}.pt'.format(finetune_epoch))
            torch.save(state, model_path)
        end_time = time.time()
        mylogger.debug('cost time is: {}'.format(end_time - start_time))
    return wholebestacc


def dummypredict(net, dce, x):
    if args.backbone == "WideResnet":
        x = net.front_end(x).unsqueeze(1)
        out = net.conv1(x)
        out = net.layer1(out)
        out = net.layer2(out)
        out = net.layer3(out)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out1 = net.clf2(out)
        out = net.linear(out)
        out = net.linear2(out)
        features_square = torch.sum(torch.pow(out, 2), 1, keepdim=True)  # x为50*2， features_square为50*1
        centers_square = torch.sum(torch.pow(dce.dummy_centers, 2), 0,
                                   keepdim=True)  # centers为2*10，则centers_square为1*10
        features_into_centers = 2 * torch.matmul(out, dce.dummy_centers)  # features_into_centers为50*10
        dist = features_square + centers_square - features_into_centers
        return -dist, out1, out


def pre2block(net, x):
    if args.backbone == "WideResnet":
        x = net.front_end(x).unsqueeze(1)
        out = net.conv1(x)
        out = net.layer1(out)
        out = net.layer2(out)
        return out


def latter2blockclf1(net, dce, x):
    if args.backbone == "WideResnet":
        out = net.layer3(x)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = net.linear(out)
        out1 = net.linear1(out)
        out = net.linear2(out)
        centers, distances = dce(out)
        return distances, out1, out


def latter2blockclf2(net, dce, x):
    if args.backbone == "WideResnet":
        out = net.layer3(x)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out1 = net.clf2(out)
        # return out
        out = net.linear(out)
        out = net.linear2(out)
        features_square = torch.sum(torch.pow(out, 2), 1, keepdim=True)  # x为50*2， features_square为50*1
        centers_square = torch.sum(torch.pow(dce.dummy_centers, 2), 0,
                                   keepdim=True)  # centers为2*10，则centers_square为1*10
        features_into_centers = 2 * torch.matmul(out, dce.dummy_centers)  # features_into_centers为50*10
        dist = features_square + centers_square - features_into_centers
        return -dist, out1


def getmodel(args):
    print('==> Building model..')
    if args.backbone == 'WideResnet':
        net = Wide_ResNet(28, 5, 0.3, args.known_class, args.feat_dim)
    net = net.cuda()
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model_type', default='Proser', type=str, help='Recognition Method')
    parser.add_argument('--backbone', default='WideResnet', type=str, help='Backbone type.')
    parser.add_argument('--dataset', default='cifar10_relabel', type=str, help='dataset configuration')
    parser.add_argument('--gpu', default='0', type=str, help='use gpu')
    parser.add_argument('--known_class', default=10, type=int, help='number of known class')
    parser.add_argument('--seed', default='9', type=int, help='random seed for dataset generation.')
    parser.add_argument('--lamda1', default='1', type=float, help='trade-off between loss')
    parser.add_argument('--lamda2', default='1', type=float, help='trade-off between loss')
    parser.add_argument('--alpha', default='3', type=float, help='alpha value for beta distribution')
    parser.add_argument('--dummynumber', default=1, type=int, help='number of dummy label.')
    parser.add_argument('--shmode', action='store_true')
    parser.add_argument('--decay', type=float, default=5e-4, help='you hua qi')
    parser.add_argument('--upsampling', action='store_true', default=False,
                        help='Do not do upsampling.')
    parser.add_argument('--classes', type=int, default=10, help='最开始有多少类')
    parser.add_argument('--feat_dim', type=int, default=4, help='原型维度')

    args = parser.parse_args()
    # pprint(vars(args))
    args.cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0

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

    nov_set = nii_dset.NIIDataSetLoader(
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

    train_dataset_loader = data_handler.IncrementalLoader(trn_set,
                                                          args.classes, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                          cuda=args.cuda, oversampling=not args.upsampling,
                                                          )
    # Loader for test data.
    test_dataset_loader = data_handler.IncrementalLoader(val_set,
                                                         args.classes,
                                                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                         cuda=args.cuda, oversampling=not args.upsampling,
                                                         )
    novel_dataset_loader = data_handler.IncrementalLoader(nov_set,
                                                          args.classes,
                                                          [15],
                                                          cuda=args.cuda, oversampling=not args.upsampling,
                                                          )

    kwargs = {'num_workers': 8, 'pin_memory': False} if args.cuda else {}
    collate_fn = nii_collate_fn.customize_collate
    # Iterator to iterate over training data.
    train_iterator = torch.utils.data.DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=True,
                                                 **kwargs)
    # Iterator to iterate over test data
    test_iterator = torch.utils.data.DataLoader(test_dataset_loader, batch_size=128, shuffle=True,
                                                **kwargs)
    novel_iterator = torch.utils.data.DataLoader(novel_dataset_loader, batch_size=128, shuffle=True,
                                                 **kwargs)

    # if args.dataset == 'cifar10_relabel':
    #     from data.cifar10_relabel import CIFAR10 as Dataset

    # trainset = Dataset('train', seed=args.seed)
    # knownlist, unknownlist = trainset.known_class_show()
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    # closeset = Dataset('testclose', seed=args.seed)
    # closerloader = torch.utils.data.DataLoader(closeset, batch_size=500, shuffle=True, num_workers=4)
    # openset = Dataset('testopen', seed=args.seed)
    # openloader = torch.utils.data.DataLoader(openset, batch_size=500, shuffle=True, num_workers=4)
    #
    # save_path1 = osp.join('results', 'D{}-M{}-B{}'.format(args.dataset, args.model_type, args.backbone, ))
    # model_path = osp.join('results', 'D{}-M{}-B{}'.format(args.dataset, 'softmax', args.backbone, ))
    # save_path2 = 'LR{}-K{}-U{}-Seed{}'.format(str(args.lr), knownlist, unknownlist, str(args.seed))
    # args.save_path = osp.join(save_path1, save_path2)
    # ensure_path(save_path1, remove=False)
    # ensure_path(args.save_path, remove=False)
    model_dir = 'proto2/proto2_l1_{}_l2_{}_a_{}_d_{}_s_{}_de_{}'.format \
        (args.lamda1, args.lamda2, args.alpha, args.dummynumber, args.seed, args.decay)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    mylogger = mylogger.mylogger().logger
    mylogger.debug(vars(args))

    globalacc = 0
    finetune_proser(59)
