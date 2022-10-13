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
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from models.wide_proto_resnet import Wide_ResNet
from models.proto_type2 import DceLoss
import copy
import random
import core_scripts.other_tools.list_tools as nii_list_tool
import core_scripts.other_tools.list_tools2019 as nii_list_tool2019
import core_scripts.data_io.default_data_io as nii_dset
import core_scripts.data_io.default_data_io2019 as nii_dset2019
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
    all_nb = 0
    all_right = 0
    for i in range(len(pred)):
        nb_per_class[target[i].detach().cpu()] += 1
        all_nb += 1
        if target[i] == pred[i]:
            correct_per_class[target[i].detach().cpu()] += 1
            all_right += 1
    # correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    for j in range(correct_per_class.shape[0]):
        mylogger.info('class: {} correct rate = 100 * {} / {} = {}'.format(j, correct_per_class[j], nb_per_class[j],
                                                                           100 * correct_per_class[j] / nb_per_class[
                                                                               j]))
    mylogger.info('all correct rate = {} / {} = {} %'.format(all_right, all_nb, 100 * all_right / all_nb))

    # return 100. * correct / len(loader.dataset)
    return nb_per_class, correct_per_class


def traindummy(epoch, net, or_net, dce):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params': net.parameters()}, {'params': dce.parameters()}], lr=args.lr * 0.01, momentum=0.9,
                          weight_decay=args.decay)

    mylogger.debug('Epoch: %d' % epoch)
    net.train()
    dce.train()
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
        index_time = 20
        while index_time > 0:
            if torch.eq(prehalflabels, prehalflabels[index]).detach().cpu().sum() != 0:
                index = torch.randperm(prehalfinputs.size(0)).cuda()
                index_time -= 1
            else:
                break
        if torch.eq(prehalflabels, prehalflabels[index]).detach().cpu().sum() != 0:
            continue
        pre2embeddings = pre2block(or_net, prehalfinputs)
        mixed_embeddings = beta * pre2embeddings + (1 - beta) * pre2embeddings[index]  # 产生虚拟数据点

        dummylogit, dummydist = dummypredict(net, laterhalfinputs, dce)  # 输出batch_size,1的向量
        lateroutputs, dist_feat = net(laterhalfinputs)  # 正常输出batch_size,6的向量
        distances = dce(dist_feat)
        latterhalfoutput = torch.cat((lateroutputs, dummylogit), 1)  # 将上面两个向量组合为batch_size，7的向量
        latterhalfdist = torch.cat((distances, dummydist), 1)
        # 将前半个batch_size的数据(经过混合)完成网络输出，并组合为batch_size，7的向量
        l2b1, dist1 = latter2blockclf1(net, dce, mixed_embeddings)
        l2b2, dist2 = latter2blockclf2(net, dce, mixed_embeddings)
        prehalfoutput = torch.cat((l2b1, l2b2), 1)
        prehalfdist = torch.cat((dist1, dist2), 1)

        maxdummy, _ = torch.max(dummylogit.clone(), dim=1)  # 对batch_size，1的向量进行softmax
        mindummydist, _ = torch.max(dummydist.clone(), dim=1)
        maxdummy = maxdummy.view(-1, 1)
        mindummydist = mindummydist.view(-1, 1)
        dummpyoutputs = torch.cat((lateroutputs.clone(), maxdummy), dim=1)  # 将batch_size,6的向量和batch_size,1的向量组合
        dummpydist = torch.cat((distances.clone(), mindummydist), dim=1)
        # 将原来的标签置信度减小
        for i in range(len(dummpyoutputs)):
            nowlabel = laterhalflabels[i]
            dummpyoutputs[i][nowlabel] = -1e9
            dummpydist[i][nowlabel] = -1e9
        dummytargets = torch.ones_like(laterhalflabels) * args.known_class

        outputs = torch.cat((prehalfdist, latterhalfdist), 0)
        # 插入的虚拟类损失
        loss1 = criterion(prehalfoutput, (torch.ones_like(prehalflabels) * args.known_class).long().cuda())
        loss1_1 = 0.5 * criterion(prehalfdist, (torch.ones_like(prehalflabels) * args.known_class).long().cuda())
        # 真实类的损失
        loss2 = criterion(latterhalfoutput, laterhalflabels)
        loss2_1 = 0.5 * criterion(latterhalfdist, laterhalflabels)
        loss2_2 = dce.regularization(dist_feat, laterhalflabels) * 0.1
        # 第二置信度损失
        loss3 = criterion(dummpyoutputs, dummytargets)
        loss3_1 = 0.5 * criterion(dummpydist, dummytargets)
        loss = 0.01 * (loss1 + loss1_1) + args.lamda1 * (loss2 + loss2_1 + loss2_2) + args.lamda2 * (loss3 + loss3_1)
        # loss = 0.01 * loss1 + args.lamda1 * loss2 + args.lamda2 * loss3

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # print(dce.centers)
        # print(dce.dummy_centers)

        # if args.shmode == False and (batch_idx + 1) % 20 == 0:
        progress_bar(batch_idx, len(train_iterator),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d) L1 %.3f, %.3f, L2 %.3f, %.3f, %.3f, L3 %.3f, %.3f,' \
                     % (
                         train_loss / (batch_idx + 1), 100. * correct / total, correct, total, loss1.item(),
                         loss1_1.item(),
                         loss2.item(), loss2_1.item(), loss2_2.item(), loss3.item(), loss3_1.item()))


def valdummy(epoch, net, dce, mainepoch):
    net.eval()
    dce.eval()
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
            logits, dist_feat = net(inputs)
            distances = dce(dist_feat)
            dummylogit, dummydist = dummypredict(net, inputs, dce)
            maxdummydist, _ = torch.max(dummydist, 1)
            maxdummydist = maxdummydist.view(-1, 1)
            totallogits = torch.cat((distances, maxdummydist), dim=1)
            closelogits[batch_idx * batchnum:batch_idx * batchnum + batchnum, :] = totallogits
        for batch_idx, (inputs, targets) in enumerate(novel_iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            # all_targets.extend(targets)
            batchnum = len(targets)
            logits, dist_feat = net(inputs)
            distances = dce(dist_feat)
            dummylogit, dummydist = dummypredict(net, inputs, dce)
            maxdummydist, _ = torch.max(dummydist, 1)
            maxdummydist = maxdummydist.view(-1, 1)
            totallogits = torch.cat((distances, maxdummydist), dim=1)
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
                    maxknownconf, _ = torch.max(embeddings[:, :-1], dim=1)
                    _, pred = torch.max(embeddings, dim=1)
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
                f1 = f1_score(closecompare / len(close_preds), opencompare / len(open_preds))
                auc1 = roc_auc_score(1 - totalbinary, totalconf)
                auc2 = roc_auc_score(totalbinary, totalconf)
                mylogger.info(
                    'Temperature: {} bias: {} AUC_by_Delta_confidence: {} f1_score: {}'.format(temperature, biasitem,
                                                                                               auc1, f1))
                auclist1.append(np.max([auc1, auc2]))
    return np.max(np.array(auclist1))


def f1_score(TP, TN):
    FN = 1 - TP
    FP = 1 - TN
    f1 = 2 * TP / (2 * TP + FN + FP)
    return f1


def my_valdummy(epoch, net, dce, mainepoch):
    net.eval()
    dce.eval()
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
    new_memory = []
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
            logits, dist_feat = net(inputs)
            distances = dce(dist_feat)
            dummylogit, dummydist = dummypredict(net, inputs, dce)
            maxdummydist, _ = torch.max(dummydist, 1)
            maxdummydist = maxdummydist.view(-1, 1)
            totallogits = torch.cat((distances, maxdummydist), dim=1)
            closelogits[batch_idx * batchnum:batch_idx * batchnum + batchnum, :] = totallogits
        for batch_idx, (inputs, targets) in enumerate(novel_iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            # all_targets.extend(targets)
            batchnum = len(targets)
            logits, dist_feat = net(inputs)
            distances = dce(dist_feat)
            dummylogit, dummydist = dummypredict(net, inputs, dce)
            maxdummydist, _ = torch.max(dummydist, 1)
            maxdummydist = maxdummydist.view(-1, 1)
            totallogits = torch.cat((distances, maxdummydist), dim=1)
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
                    maxknownconf, _ = torch.max(embeddings[:, :-1], dim=1)
                    _, pred = torch.max(embeddings, dim=1)
                    all_preds.extend(pred)
                    maxknownconf = maxknownconf.view(-1, 1)
                    conf = dummyconf - maxknownconf
                    for idx, c in enumerate(conf):
                        if c < 0:
                            close_preds.append(1)
                        else:
                            close_preds.append(0)
                            # start = test_iterator.dataset.indices[10][0]
                            # new_memory.append(test_iterator.dataset.data[start + batch_idx * Logitsbatchsize + idx])
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
                    for idx, c in enumerate(conf):
                        if c > 0:
                            open_preds.append(0)
                            start = novel_iterator.dataset.indices[args.novel[0]][0]
                            new_memory.append(novel_iterator.dataset.data[start + batch_idx * Logitsbatchsize + idx])
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
                start = train_iterator.dataset.indices[args.novel[0]][0]
                train_iterator.dataset.data[start:start + 200] = new_memory[:200]
                train_iterator.dataset.indices[args.novel[0]] = (train_iterator.dataset.indices[args.novel[0]][0],
                                                                 train_iterator.dataset.indices[args.novel[0]][0] + 200)
    return np.max(np.array(auclist1))


import matplotlib.pyplot as mplt


def my_plot(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = mplt.figure()
    ax = mplt.subplot(111)
    for i in range(data.shape[0]):
        mplt.text(data[i, 0], data[i, 1], str(label[i]),
                  color=mplt.cm.Set1(label[i] / 10.),
                  fontdict={'weight': 'bold', 'size': 9})
    mplt.xticks([])
    mplt.yticks([])
    mplt.title(title)
    return fig


def plot_embeding(net, dce):
    net.eval()
    dce.eval()
    tss = []
    tgg = []
    ii = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_iterator):
            inputs = inputs.cuda()
            em = net(inputs)
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            t0 = time.time()
            ts = em.detach().cpu().numpy().reshape(args.batch_size, 320 * 12)
            result = tsne.fit_transform(ts)
            fig = my_plot(result, targets.numpy(),
                          't-SNE embedding of the digits (time %.2fs)'
                          % (time.time() - t0))
            mplt.show()
            # tg = targets.numpy()
            # tss.append(ts)
            # tgg.append(tg)
            # ii = 32 * (batch_idx + 1)
            # if ii > 500:
            #     break
    #             break
    # tss = np.array(tss).reshape(ii, -1)
    # targets = np.array(tgg).reshape(-1)
    # result = tsne.fit_transform(tss)
    # fig = my_plot(result, targets,
    #               't-SNE embedding of the digits (time %.2fs)'
    #               % (time.time() - t0))
    # mplt.show()


def my_finetune_proser(epoch=59):
    print('Now processing epoch', epoch)
    net = getmodel(args)
    print('==> Resuming from checkpoint..')
    # assert os.path.isdir(model_path), 'Error: no checkpoint directory found!'
    modelname = 'Modelof_Epoch' + str(epoch) + '.pth'
    checkpoint = torch.load('/home/hadoop/桌面/代码/新颖性检测/latest-CVPR21-Proser-main/CVPR21-Proser-main/logger/results'
                            '/proser2_l1_1.0_l2_7.0_a_3.0_d_1_s_9_de_0.009/4.pt')
    net.clf2 = nn.Linear(3840, args.dummynumber)
    net.load_state_dict(checkpoint['net'])

    # net.clf2 = nn.Linear(3840, args.dummynumber)
    # 双卡
    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         net = torch.nn.DataParallel(net,device_ids=[0,1])
    net = net.cuda()
    or_net = copy.deepcopy(net)
    for param in or_net.parameters():
        param.requires_grad = False
    dce = DceLoss(args.known_class, args.feat_dim)
    # dce.load_state_dict(checkpoint['dce'])
    dce.dummy_centers = torch.nn.Parameter(torch.randn(args.feat_dim, args.dummynumber).cuda(), requires_grad=True)
    dce.load_state_dict(checkpoint['dce'])
    dce.cuda()

    # checkpoint2 = torch.load(
    #     '/home/hadoop/桌面/代码/中文伪造音频检测/incremental-learning-autoencoders_detect_novel-icral/model_save_中文增量_decay_0.005_seed_12345_class_13/model_0_epoch_4.pt')
    # train_iterator.dataset.data = checkpoint2['data']
    # train_iterator.dataset.indices = checkpoint2['indices']

    # plot_embeding(net, dce)
    finetuneacc = my_valdummy(0, net, dce, epoch)
    # state = {
    #     'net': net.state_dict(),
    #     'dce': dce.state_dict(),
    #     'data': train_iterator.dataset.data,
    #     'indices': train_iterator.dataset.indices,
    #     'epoch': epoch,
    # }
    # dir = 'proto3_1/proser2_l1_{}_l2_{}_a_{}_d_{}_s_{}_de_{}_n_{}'.format \
    #     (args.lamda1, args.lamda2, args.alpha, args.dummynumber, args.seed, args.decay, args.classes)
    # if not os.path.exists(dir):
    #     os.makedirs(dir)
    # path = os.path.join(dir, 'increment.pt')
    # torch.save(state, path)
    return finetuneacc


def finetune_proser(epoch=59):
    print('Now processing epoch', epoch)
    net = getmodel(args)
    print('==> Resuming from checkpoint..')
    # assert os.path.isdir(model_path), 'Error: no checkpoint directory found!'
    modelname = 'Modelof_Epoch' + str(epoch) + '.pth'
    checkpoint = torch.load(
        '/home/hadoop/桌面/代码/中文伪造音频检测/incremental-learning-autoencoders_detect_novel-icral/model_save_中文增量_decay_0.005_seed_12345_class_10/model_0_epoch_4.pt')
    # net.clf2 = nn.Linear(3840, args.dummynumber)
    net.load_state_dict(checkpoint['net'], strict=False)

    net.clf2 = nn.Linear(3840, args.dummynumber)
    checkpoint2 = torch.load(
        '/home/hadoop/桌面/代码/中文伪造音频检测/latest-CVPR21-Proser-main/CVPR21-Proser-main/proto3_1/proser2_l1_2.0_l2_1.0_a_5.0_d_1_s_9_de_0.005_n_12/increment.pt')
    # net.clf2.weight.data = checkpoint2['net']['clf2.weight'].data
    # net.clf2.bias.data = checkpoint2['net']['clf2.bias'].data

    # # 双卡
    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         net = torch.nn.DataParallel(net)

    net = net.cuda()
    or_net = copy.deepcopy(net)
    for param in or_net.parameters():
        param.requires_grad = False
    dce = DceLoss(args.known_class, args.feat_dim)
    dce.load_state_dict(checkpoint['dce'])
    dce.dummy_centers = torch.nn.Parameter(torch.randn(args.feat_dim, args.dummynumber).cuda(), requires_grad=True)
    # dce.dummy_centers.data = checkpoint2['dce']['dummy_centers'].data
    # dce.load_state_dict(checkpoint['dce'])
    dce.cuda()
    train_iterator.dataset.data = checkpoint2['data']
    train_iterator.dataset.indices = checkpoint2['indices']
    train_dataset_loader.update_length()

    FineTune_MAX_EPOCH = 60
    wholebestacc = 0
    # finetuneacc = valdummy(0, net, dce, epoch)

    for finetune_epoch in range(FineTune_MAX_EPOCH):
        start_time = time.time()
        traindummy(finetune_epoch, net, or_net, dce)
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


def dummypredict(net, x, dce):
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
        feat = net.linear(out)
        feat = net.linear2(feat)
        dist = dce.dummy_f(feat)
        return out1, dist

def double_pre2block(net, x):
    if args.backbone == "WideResnet":
        x = net.module.front_end(x).unsqueeze(1)
        out = net.module.conv1(x)
        out = net.module.layer1(out)
        out = net.module.layer2(out)

        # out = net.module.layer3(out)
        # out = F.relu(net.module.bn1(out))
        # out = F.avg_pool2d(out, 8)
        return out

def pre2block(net, x):
    if args.backbone == "WideResnet":
        x = net.front_end(x).unsqueeze(1)
        out = net.conv1(x)
        out = net.layer1(out)
        out = net.layer2(out)

        # out = net.layer3(out)
        # out = F.relu(net.bn1(out))
        # out = F.avg_pool2d(out, 8)
        return out


def latter2blockclf1(net, dce, x):
    if args.backbone == "WideResnet":
        out = net.layer3(x)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = net.linear(out)
        out1 = net.linear1(out)
        feat = net.linear2(out)
        dist = dce(feat)
        return out1, dist


def latter2blockclf2(net, dce, x):
    if args.backbone == "WideResnet":
        out = net.layer3(x)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out1 = net.clf2(out)
        feat = net.linear(out)
        feat = net.linear2(feat)
        dist = dce.dummy_f(feat)
        return out1, dist


def getmodel(args):
    print('==> Building model..')
    if args.backbone == 'WideResnet':
        net = Wide_ResNet(28, 5, 0.3, args.known_class, args.feat_dim)
    net = net.cuda()
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model_type', default='Proser', type=str, help='Recognition Method')
    parser.add_argument('--backbone', default='WideResnet', type=str, help='Backbone type.')
    parser.add_argument('--dataset', default='cifar10_relabel', type=str, help='dataset configuration')
    parser.add_argument('--gpu', default='1', type=str, help='use gpu')
    parser.add_argument('--known_class', default=10, type=int, help='number of known class')
    parser.add_argument('--seed', default='9', type=int, help='random seed for dataset generation.')
    parser.add_argument('--lamda1', default='2', type=float, help='trade-off between loss')
    parser.add_argument('--lamda2', default='2', type=float, help='trade-off between loss')
    parser.add_argument('--alpha', default='5', type=float, help='alpha value for beta distribution')
    parser.add_argument('--dummynumber', default=1, type=int, help='number of dummy label.')
    parser.add_argument('--shmode', action='store_true')
    parser.add_argument('--decay', type=float, default=0.005, help='you hua qi')
    parser.add_argument('--upsampling', action='store_true', default=False,
                        help='Do not do upsampling.')
    parser.add_argument('--classes', type=int, default=10, help='最开始有多少类')
    parser.add_argument('--feat_dim', type=int, default=4, help='原型维度')
    parser.add_argument('--novel', nargs='+', type=int, default=[10])

    args = parser.parse_args()
    # pprint(vars(args))
    args.cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0
    start_epoch = 0

    print('==> Preparing data..')

    prj_conf = importlib.import_module('config2019')

    # Loader used for training data

    # Load file list and create data loader
    trn_lst = nii_list_tool2019.read_list_from_text(prj_conf.proto_train_dir)
    trn_set = nii_dset2019.NIIDataSetLoader(
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

    val_lst = nii_list_tool2019.read_list_from_text(prj_conf.proto_dev_dir)
    val_set = nii_dset2019.NIIDataSetLoader(
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

    nov_lst = nii_list_tool.read_list_from_text(prj_conf.test_list)
    nov_set = nii_dset.NIIDataSetLoader(
        'dev',
        args,
        transforms,
        prj_conf.test_eval_set_name,
        nov_lst,
        prj_conf.test_input_dirs,
        prj_conf.test_list,
        prj_conf.test_input_exts,
        prj_conf.test_input_dims,
        prj_conf.test_input_reso,
        prj_conf.test_input_norm,
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

    # nov_set = nii_dset.NIIDataSetLoader(
    #     'train',
    #     args,
    #     transforms,
    #     prj_conf.trn_eval_set_name,
    #     trn_lst,
    #     prj_conf.input_dirs,
    #     prj_conf.proto_train_dir,
    #     prj_conf.input_exts,
    #     prj_conf.input_dims,
    #     prj_conf.input_reso,
    #     prj_conf.input_norm,
    #     prj_conf.output_dirs,
    #     prj_conf.output_exts,
    #     prj_conf.output_dims,
    #     prj_conf.output_reso,
    #     prj_conf.output_norm,
    #     './',
    #     truncate_seq=prj_conf.truncate_seq,
    #     min_seq_len=prj_conf.minimum_len,
    #     save_mean_std=True,
    #     wav_samp_rate=prj_conf.wav_samp_rate)

    train_dataset_loader = data_handler.IncrementalLoader(trn_set,
                                                          args.classes,
                                                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
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
                                                          args.novel,
                                                          cuda=args.cuda, oversampling=not args.upsampling,
                                                          )

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    collate_fn = nii_collate_fn.customize_collate
    # Iterator to iterate over training data.
    train_iterator = torch.utils.data.DataLoader(train_dataset_loader, batch_size=args.batch_size, shuffle=True,
                                                 **kwargs)
    # Iterator to iterate over test data
    test_iterator = torch.utils.data.DataLoader(test_dataset_loader, batch_size=128, shuffle=False,
                                                **kwargs)
    novel_iterator = torch.utils.data.DataLoader(novel_dataset_loader, batch_size=128, shuffle=False,
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
    model_dir = 'proto3/proser_l1_{}_l2_{}_a_{}_d_{}_s_{}_de_{}_n_{}'.format \
        (args.lamda1, args.lamda2, args.alpha, args.dummynumber, args.seed, args.decay, args.classes)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mylogger = mylogger.mylogger().logger
    mylogger.debug(vars(args))

    globalacc = 0
    my_finetune_proser(59)
