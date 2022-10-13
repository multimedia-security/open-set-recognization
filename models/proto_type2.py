import torch.nn as torch_nn
import torch
import torch.nn.functional as F


class DceLoss(torch_nn.Module):
    def __init__(self, n_classes, feat_dim, init_weight=True):
        super(DceLoss, self).__init__()
        self.add_center = None
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.centers = torch_nn.Parameter(torch.randn(self.feat_dim, self.n_classes).cuda(), requires_grad=True)
        # self.radius = torch_nn.Parameter(torch.zeros(self.n_classes), requires_grad=True)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        torch_nn.init.kaiming_normal_(self.centers)

    # def modified_radius(self, nb_add=1):
    #     add_radius = torch.zeros(nb_add).cuda()
    #     radius = torch.cat((self.radius, add_radius), dim=0)
    #     self.radius = torch.nn.Parameter(radius, requires_grad=True)

    def regularization(self, features, labels):
        p = torch.t(self.centers)[labels]
        distance = F.mse_loss(features, p)
        return distance

    def modefied_centers(self, nb_add=1):
        add_center = torch.randn(self.feat_dim, nb_add).cuda()
        add_center = torch_nn.init.kaiming_normal_(add_center)
        centers = torch.cat((self.centers, add_center), dim=1)
        self.centers = torch_nn.Parameter(centers, requires_grad=True)

    def dummy_f(self, out):
        features_square = torch.sum(torch.pow(out, 2), 1, keepdim=True)  # x为50*2， features_square为50*1
        centers_square = torch.sum(torch.pow(self.dummy_centers, 2), 0,
                                   keepdim=True)  # centers为2*10，则centers_square为1*10
        features_into_centers = 2 * torch.matmul(out, self.dummy_centers)  # features_into_centers为50*10
        dist = features_square + centers_square - features_into_centers
        return -dist

    def forward(self, x):
        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)  # x为50*2， features_square为50*1
        centers_square = torch.sum(torch.pow(self.centers, 2), 0, keepdim=True)  # centers为2*10，则centers_square为1*10
        features_into_centers = 2 * torch.matmul(x, self.centers)  # features_into_centers为50*10
        dist = features_square + centers_square - features_into_centers
        return -dist
