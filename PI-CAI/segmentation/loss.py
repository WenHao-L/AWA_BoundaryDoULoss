import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance


class FocalLoss(nn.Module):
    """Focal loss function for binary segmentation."""

    def __init__(self, alpha=1, gamma=2, num_classes=2, reduction="sum"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs,dim=1)
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

        p_t = (inputs * targets) + ((1 - inputs) * (1 - targets))
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class Deep_Supervised_Loss(nn.Module):
    def __init__(self):
        super(Deep_Supervised_Loss, self).__init__()
        self.loss = FocalLoss()
    def forward(self, input, target):
        loss = 0
        # print(type(input))
        for i, img in enumerate(input):
            w = 1 / (2 ** i)
            label = F.interpolate(target, img.size()[2:])
            l = self.loss(img, label)
            loss += l * w
        return loss 
    

class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes=2):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes


class BDoU_Deep_Supervised_Loss(nn.Module):
    def __init__(self):
        super(BDoU_Deep_Supervised_Loss, self).__init__()
        self.loss = BoundaryDoULoss()

    def forward(self, input, target):
        loss = 0
        # print(type(input))
        for i, img in enumerate(input):
            w = 1 / (2 ** i)
            label = F.interpolate(target, img.size()[2:])
            l = self.loss(img, label)
            loss += l * w
        return loss 
    

class TverskyLoss(nn.Module):
    def __init__(self, classes=2) -> None:
        super().__init__()
        self.classes = classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, y_pred, y_true, alpha=0.7, beta=0.3):

        y_pred = torch.softmax(y_pred, dim=1)
        # y_true = self._one_hot_encoder(y_true)
        loss = 0
        for i in range(1, self.classes):
            p0 = y_pred[:, i, :, :]
            ones = torch.ones_like(p0)
            #p1: prob that the pixel is of class 0
            p1 = ones - p0  
            g0 = y_true[:, i, :, :]
            g1 = ones - g0
            #terms in the Tversky loss function combined with weights
            tp = torch.sum(p0 * g0)
            fp = alpha * torch.sum(p0 * g1)
            fn = beta * torch.sum(p1 * g0)
            #add to the denominator a small epsilon to prevent the value from being undefined 
            EPS = 1e-5
            num = tp
            den = tp + fp + fn + EPS
            result = num / den
            loss += result
        return 1 - loss / self.classes


class Tversky_Deep_Supervised_Loss(nn.Module):
    def __init__(self):
        super(Tversky_Deep_Supervised_Loss, self).__init__()
        self.loss = TverskyLoss()

    def forward(self, input, target):
        loss = 0
        # print(type(input))
        for i, img in enumerate(input):
            w = 1 / (2 ** i)
            label = F.interpolate(target, img.size()[2:])
            l = self.loss(img, label)
            loss += l * w
        return loss 
    

class BoundaryLoss(nn.Module):
    def __init__(self, classes=2) -> None:
        super().__init__()
        # ignore background
        self.idx = [i for i in range(1, classes)]

    def compute_sdf1_1(self, img_gt, out_shape):
        img_gt = img_gt.cpu().numpy().astype(np.uint8)

        normalized_sdf = np.zeros(out_shape)

        for b in range(out_shape[0]): # batch size
            # ignore background
            for c in range(1, out_shape[1]):
                posmask = img_gt[b, c].astype(np.bool_)
                if posmask.any():
                    negmask = ~posmask
                    posdis = distance(posmask)
                    negdis = distance(negmask)
                    # sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) * negmask - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis)) * posmask
                    sdf = negdis * negmask - (posdis -1) * posmask
                    normalized_sdf[b][c] = sdf

        return normalized_sdf

    def forward(self, outputs, gt):
        outputs_soft = F.softmax(outputs, dim=1)
        gt_sdf = self.compute_sdf1_1(gt, outputs_soft.shape)
        pc = outputs_soft[:,self.idx,...]
        dc = torch.from_numpy(gt_sdf[:,self.idx,...]).cuda()
        multipled = torch.einsum('bcwh, bcwh->bcwh', pc, dc)
        bd_loss = multipled.mean()

        return bd_loss


class Boundary_Deep_Supervised_Loss(nn.Module):
    def __init__(self):
        super(Boundary_Deep_Supervised_Loss, self).__init__()
        self.bou_loss = BoundaryLoss()
        self.focal_loss = FocalLoss()

    def forward(self, input, target):                                                                                                                        
        loss = 0
        for i, img in enumerate(input):
            w = 1 / (2 ** i)
            label = F.interpolate(target, img.size()[2:])
            l = self.focal_loss(img, label) + 0.01 * self.bou_loss(img, label)
            loss += l * w
        return loss
