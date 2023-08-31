''' Incremental-Classifier Learning 
 '''

from __future__ import print_function

import copy
import logging
import os.path
from ndnet.detector.CSND import CSND
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from model.resnet32 import modified_linear
from models.wide_proto_resnet import Wide_ResNet
from models.proto_type import DceLoss

import model
import sandbox.util_frontend as nii_front_end

logger = logging.getLogger('iCARL')


class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator=None):
        self.train_data_iterator = trainDataIterator
        self.test_data_iterator = testDataIterator
        self.model = model
        self.args = args
        self.dataset = dataset
        self.train_loader = self.train_data_iterator.dataset
        self.older_classes = []
        self.optimizer = optimizer
        self.model_fixed = copy.deepcopy(self.model)
        self.active_classes = []
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        self.models = []
        self.current_lr = args.lr
        # self.all_classes = list(range(dataset.classes))
        # self.all_classes.sort(reverse=True)
        self.left_over = []
        self.ideal_iterator = ideal_iterator
        self.model_single = copy.deepcopy(self.model)
        self.optimizer_single = None

        logger.warning("Shuffling turned off for debugging")
        # random.seed(args.seed)
        # random.shuffle(self.all_classes)


class Trainer(GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, dce, args, optimizer, ideal_iterator=None):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dce_fixed = None
        self.args = args
        self.dce = dce
        if os.path.exists('model_save/model_epoch_sum_4.pt'):
            f_sum = torch.load('model_save/model_epoch_sum_4.pt')
            self.f_sum_max_all = f_sum['f_sum_max_all']
            self.f_sum_max_all.requires_grad = False
            self.f_sum_sec_all = f_sum['f_sum_sec_all']
            self.f_sum_sec_all.requires_grad = False
            self.nb_correct = f_sum['nb_correct']
            self.nb_correct.requires_grad = False
        else:
            self.f_sum_max_all = torch.zeros(self.args.classes).cuda()
            self.f_sum_sec_all = torch.zeros(self.args.classes).cuda()
            self.nb_correct = torch.zeros(self.args.classes).cuda() + 0.0001

    def reset_sum(self):
        self.f_sum_max_all = self.f_sum_max_all.zero_()
        self.f_sum_sec_all = self.f_sum_sec_all.zero_()
        self.nb_correct = self.nb_correct.zero_() + 0.0001

    def update_lr(self, epoch):
        for temp in range(0, len(self.args.schedule)):
            if self.args.schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    logger.debug("Changing learning rate from %0.5f to %0.5f", self.current_lr,
                                 self.current_lr * self.args.gammas[temp])
                    self.current_lr *= self.args.gammas[temp]

    def modified_model(self, class_group, add_fc=1):
        out_features = self.model.fc.out_features
        in_features = self.model.fc.in_features
        new_fc = modified_linear(in_features, out_features, add_fc)
        if class_group == 1:
            new_fc.fc1.weight.data = self.model.fc.weight.data
        else:
            new_fc.fc1.weight.data[:self.model.fc.fc1.out_features] = self.model.fc.fc1.weight.data
            new_fc.fc1.weight.data[self.model.fc.fc1.out_features:] = self.model.fc.fc2.weight.data
        self.model.fc = new_fc.cuda()
        self.model_fixed.fc = new_fc.cuda()
        self.optimizer = torch.optim.SGD([{'params': self.model.parameters()}, {'params': self.dce.parameters()}],
                                         self.args.lr, momentum=self.args.momentum,
                                         weight_decay=self.args.decay, nesterov=True)
        # self.optimizer_dce = torch.optim.SGD(self.dce.parameters(), self.args.lr, momentum=self.args.momentum,
        #                                      weight_decay=self.args.decay, nesterov=True)
        # self.model.cuda()
        # self.model_fixed.cuda()

    def increment_classes(self, class_group, add_classes):
        '''
        Add classes starting from class_group to class_group + step_size 
        :param class_group: 
        :return: N/A. Only has side-affects 
        '''
        for add_class in add_classes[class_group]:
            # pop_val = self.all_classes.pop()
            pop_val = add_class
            self.train_data_iterator.dataset.add_class(pop_val)
            # self.ideal_iterator.dataset.add_class(pop_val)
            self.test_data_iterator.dataset.add_class(pop_val)
            self.left_over.append(pop_val)

    def limit_class(self, n, k, cur_len, herding=True):
        if not herding:
            self.train_loader.limit_class(n, k)
        else:
            self.train_loader.limit_class_and_sort(n, k, self.model_fixed, cur_len)
        if n not in self.older_classes:
            self.older_classes.append(n)

    def setup_training(self, cur_len):
        for param_group in self.optimizer.param_groups:
            logger.debug("Setting LR to %0.2f", self.args.lr)
            param_group['lr'] = self.args.lr
            self.current_lr = self.args.lr
        for val in self.left_over:
            # self.limit_class(val, int(self.args.memory_budget / len(self.left_over)), not self.args.no_herding)
            self.limit_class(val, int(self.args.memory_budget), cur_len, not self.args.no_herding)

    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        self.dce_fixed = copy.deepcopy(self.dce)
        self.dce_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        for param in self.dce_fixed.parameters():
            param.requires_grad = False
        self.models.append(self.model_fixed)

        if self.args.random_init:
            logger.warning("Random Initilization of weights at each increment")
            myModel = Wide_ResNet(28, 5, 0.3, self.args.known_class, self.args.feat_dim)
            if self.args.cuda:
                myModel.cuda()
            self.model = myModel
            self.optimizer = torch.optim.SGD([{'params': self.model.parameters()}, {'params': self.dce.parameters()}],
                                             self.args.lr, momentum=self.args.momentum,
                                             weight_decay=self.args.decay, nesterov=True)
            self.model.eval()

    def randomly_init_model(self):
        logger.info("Randomly initilizaing model")
        myModel = model.ModelFactory.get_model(self.args.model_type, self.args.dataset)
        if self.args.cuda:
            myModel.cuda()
        self.model = myModel
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum,
                                         weight_decay=self.args.decay, nesterov=True)
        self.model.eval()

    def get_model(self):
        myModel = model.ModelFactory.get_model(self.args.model_type, self.args.dataset)
        if self.args.cuda:
            myModel.cuda()
        optimizer = torch.optim.SGD(myModel.parameters(), self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.decay, nesterov=True)
        myModel.eval()

        self.current_lr = self.args.lr

        self.model_single = myModel
        self.optimizer_single = optimizer

    def regularization(self, features, centers, labels):
        p = torch.t(centers)[labels]
        distance = F.mse_loss(features, p)

        return distance

    def computer_loss3(self, batch_size, target, outputs, pred, radius):
        correct_index = np.zeros(batch_size)
        for index, (tar, p) in enumerate(zip(target, pred)):
            if tar == p:
                correct_index[index] = 1
        nb_correct = np.argwhere(correct_index == 1).shape[0]
        if nb_correct == 0:
            return 0
        correct_idx = [r == 1 for r in correct_index]
        correct_targets = target[correct_idx]
        correct_outputs = outputs[correct_idx]
        radius_colum = torch.zeros(nb_correct)
        outputs_colum = torch.zeros(nb_correct)
        for idx, tar in enumerate(correct_targets):
            radius_colum[idx] = radius[tar]
            outputs_colum[idx] = torch.max(correct_outputs[idx])
        loss = F.mse_loss(radius_colum.cuda(), outputs_colum.cuda())
        return loss

    def updata_criterion(self, class_group, add_class):
        last_all = 0
        ce = []
        if class_group == 0:
            # weight_CE = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 15]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            for i in range(class_group):
                last_all += len(add_class[i])
            for _ in range(last_all):
                ce.append(15)
            for _ in range(len(add_class[class_group])):
                ce.append(1)
            weight_CE = torch.FloatTensor(ce).cuda()
            print('weight_CE: {}'.format(weight_CE))
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight_CE)

    def train(self, epoch, class_group, add_class):

        self.model.train()
        self.dce.train()
        logger.info("Epochs %d", epoch)
        losses = 0
        for idx, (data, target) in enumerate(self.train_data_iterator):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()

            outputs1, outputs2 = self.model(data)
            centers, distance = self.dce(outputs2)
            loss1 = self.criterion(outputs1, target)
            loss2_1 = self.dce.regularization(outputs2, target)
            loss2_2 = self.criterion(distance, target)
            loss = loss1 + (loss2_1 * 0.1 + loss2_2) * 0.1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses += loss.item()
            if idx % 20 == 0:
                logger.info('epoch:{} {}/{}     loss: {}  loss2_1:{}  loss2_2: {}'.format
                            (epoch, idx, len(self.train_data_iterator), losses / (idx + 1), loss2_1, loss2_2))
        logger.info('center: {}'.format(self.dce.centers))

    def add_model(self):
        model = copy.deepcopy(self.model_single)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models.append(model)
        logger.debug("Total Models %d", len(self.models))
