import torch
import torch.nn as nn


class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoULoss3D, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
        padding_out = torch.zeros((target.shape[0], target.shape[-3] + 2, target.shape[-2] + 2, target.shape[-1] + 2))
        padding_out[:, 1:-1, 1:-1, 1:-1] = target
        d, h, w = 3, 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - d + 1, padding_out.shape[2] - h + 1,

                        padding_out.shape[3] - w + 1)).cuda(target.device.index)
        for i in range(Y.shape[0]):
            Y[i, :, :,:] = torch.conv3d(target[i].unsqueeze(0),kernel.unsqueeze(0).cuda(target.device.index), padding=1)
        Y = Y * target
        Y[Y == 7] = 0
        S = torch.count_nonzero(Y)  # surface area
        V = torch.count_nonzero(target)  # volume
        smooth = 1e-5
        alpha = 1 - (S + smooth) / (V + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)
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
