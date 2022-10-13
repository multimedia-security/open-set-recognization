''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchnet.meter import confusionmeter
import sandbox.util_frontend as nii_front_end
# from model.resnet32 import DceLoss

logger = logging.getLogger('iCARL')


class EvaluatorFactory():
    '''
    This class is used to get different versions of evaluators
    '''

    def __init__(self):
        pass

    @staticmethod
    def get_evaluator(testType="nmc", args=None):
        if testType == "nmc":
            return NearestMeanEvaluator(args)
        if testType == "trainedClassifier":
            return softmax_evaluator(args)


class NearestMeanEvaluator():
    '''
    Nearest Class Mean based classifier. Mean embedding is computed and stored; at classification time, the embedding closest to the 
    input embedding corresponds to the predicted class.
    '''

    def __init__(self, args):
        self.args = args
        self.cuda = args.cuda
        self.means = None
        self.totalFeatures = np.zeros((args.classes, 1))

    def evaluate(self, model, loader, step_size=10, kMean=False):
        '''
        :param model: Train model
        :param loader: Data loader
        :param step_size: Step size for incremental learning
        :param kMean: Doesn't work very well so don't use; Will be removed in future versions 
        :return: 
        '''
        model.eval()
        if self.means is None:
            self.means = np.zeros((self.args.classes, model.featureSize))
        correct = 0
        correct_per_class = np.zeros(self.args.classes)
        nb_per_class = np.zeros(self.args.classes)
        # correct_per = np.zeros(7)
        for data, y, target in loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
                self.means = self.means.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data, True).unsqueeze(1)
            result = (output.data - self.means.float())

            result = torch.norm(result, 2, 2)
            if kMean:
                result = result.cpu().numpy()
                tempClassifier = np.zeros((len(result), int(self.args.classes / step_size)))
                for outer in range(0, len(result)):
                    for tempCounter in range(0, int(self.args.classes / step_size)):
                        tempClassifier[outer, tempCounter] = np.sum(
                            result[tempCounter * step_size:(tempCounter * step_size) + step_size])
                for outer in range(0, len(result)):
                    minClass = np.argmin(tempClassifier[outer, :])
                    result[outer, 0:minClass * step_size] += 300000
                    result[outer, minClass * step_size:(minClass + 1) * step_size] += 300000
                result = torch.from_numpy(result)
                if self.cuda:
                    result = result.cuda()
            _, pred = torch.min(result, 1)
            # print(pred.eq(target.data.view_as(pred)).cpu())
            for i in range(pred.shape[0]):
                nb_per_class[target.data[i].detach().cpu()] += 1
                if target.data[i] == pred[i]:
                    correct_per_class[target.data[i].detach().cpu()] += 1
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        for j in range(correct_per_class.shape[0]):
            logger.info('class: {} correct rate = 100 * {} / {} = {}'.format(j, correct_per_class[j], nb_per_class[j],
                                                                             100 * correct_per_class[j] / nb_per_class[
                                                                                 j]))

        return 100. * correct / len(loader.dataset)

    def get_confusion_matrix(self, model, loader, size):
        '''
        
        :param model: Trained model
        :param loader: Data iterator
        :param size: Size of confusion matrix (Equal to largest possible label predicted by the model)
        :return: 
        '''
        model.eval()
        test_loss = 0
        correct = 0
        # Get the confusion matrix object
        cMatrix = confusionmeter.ConfusionMeter(size, True)

        for data, y, target in loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
                self.means = self.means.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data, True).unsqueeze(1)
            result = (output.data - self.means.float())

            result = torch.norm(result, 2, 2)
            # NMC for classification
            _, pred = torch.min(result, 1)
            # Evaluate results
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            # Add the results in appropriate places in the matrix.
            cMatrix.add(pred, target.data.view_as(pred))

        test_loss /= len(loader.dataset)
        # Get 2d numpy matrix to remove the dependency of other code on confusionmeter
        img = cMatrix.value()
        return img

    def update_means(self, model, train_loader, classes=100):
        '''
        This method updates the mean embedding using the train data; DO NOT pass test data iterator to this. 
        :param model: Trained model
        :param train_loader: data iterator
        :param classes: Total number of classes
        :return: 
        '''
        # Set the mean to zero
        if self.means is None:
            self.means = np.zeros((classes, model.featureSize))
        self.means *= 0
        self.classes = classes
        self.means = np.zeros((classes, model.featureSize * 6))
        self.totalFeatures = np.zeros((classes, 1)) + .001
        logger.debug("Computing means")
        # Iterate over all train Dataset
        for batch_id, (data, target) in enumerate(train_loader):
            # Get features for a minibactch
            if self.cuda:
                data = data.cuda()
            features = model.forward(Variable(data), True)
            # Convert result to a numpy array
            featuresNp = features.data.cpu().numpy()
            # Accumulate the results in the means array
            # print (self.means.shape,featuresNp.shape)
            np.add.at(self.means, target, featuresNp)
            # Keep track of how many instances of a class have been seen. This should be an array with all elements = classSize
            np.add.at(self.totalFeatures, target, 1)

        # Divide the means array with total number of instances to get the average
        # print ("Total instances", self.totalFeatures)
        self.means = self.means / self.totalFeatures
        self.means = torch.from_numpy(self.means)
        # Normalize the mean vector
        self.means = self.means / torch.norm(self.means, 2, 1).unsqueeze(1)
        self.means[self.means != self.means] = 0
        self.means = self.means.unsqueeze(0)

        logger.debug("Mean vectors computed")
        # Return
        return


class softmax_evaluator():
    '''
    Evaluator class for softmax classification 
    '''

    def __init__(self, args):
        self.args = args
        self.cuda = args.cuda
        self.means = None
        self.totalFeatures = np.zeros((args.classes, 1))
        # self.dce = nii_front_end.DceLoss(self.args.classes, 4).cuda()

    def evaluate(self, model, loader, dce, scale=None, thres=False, older_classes=None, step_size=10, descriptor=False,
                 falseDec=False, higher=False):
        '''
        :param model: Trained model
        :param loader: Data iterator
        :param scale: Scale vector computed by dynamic threshold moving
        :param thres: If true, use scaling
        :param older_classes: Will be removed in next iteration
        :param step_size: Step size for incremental learning
        :param descriptor: Will be removed in next iteration; used to compare the results with a recent paper by Facebook. 
        :param falseDec: Will be removed in the next iteration.
        :param higher: Hierarchical softmax support
        :return: 
        '''
        model.eval()
        dce.eval()
        with torch.no_grad():
            correct = 0
            tempCounter = 0
            correct_per_class = np.zeros(20)
            nb_per_class = np.zeros(20)
            for data, target in loader:
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                features1, features2 = model(data)
                centers, distance = dce(features2)
                outputs = F.log_softmax(distance, dim=1)
                # output = F.log_softmax(distance, dim=1)
                _, pred = torch.max(outputs, 1)  # get the index of the max log-probability
                for i in range(pred.shape[0]):
                    nb_per_class[target.data[i].detach().cpu()] += 1
                    if target.data[i] == pred[i]:
                        correct_per_class[target.data[i].detach().cpu()] += 1
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        for j in range(correct_per_class.shape[0]):
            logger.info('class: {} correct rate = 100 * {} / {} = {}'.format(j, correct_per_class[j], nb_per_class[j],
                                                                             100 * correct_per_class[j] / nb_per_class[
                                                                                 j]))

        return 100. * correct / len(loader.dataset)

    def get_confusion_matrix(self, model, loader, size, dce):
        '''
        :return: Returns the confusion matrix on the data given by loader
        '''
        model.eval()
        dce.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0
            # Initialize confusion matrix
            cMatrix = confusionmeter.ConfusionMeter(size, True)

            # Iterate over the data and stores the results in the confusion matrix
            for data, target in loader:
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                features1, features2 = model(data)
                centers, distance = dce(features2)
                output = F.log_softmax(distance, dim=1)

                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                cMatrix.add(pred.squeeze(), target.data.view_as(pred).squeeze())

        # Returns normalized matrix.
        test_loss /= len(loader.dataset)
        img = cMatrix.value()
        return img
