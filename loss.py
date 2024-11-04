import torch
import torch.nn as nn
import torch.nn.functional as F
def clip_by_value(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """

    result = (t >= t_min)* t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max)* t_max
    return result

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False,
                 label_smooth=None, class_num=4):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.label_smooth = label_smooth
        self.class_num = class_num
        # self.pos_weight = weight[1] / weight[0]

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'focal_BCE':
            return self.Focal_BCE
        elif mode == 'ppa':
            # print('PPA loss')
            return self.PPA_loss
        elif mode == 'dice':
            return self.Dice_loss
        elif mode == 'dice_1by1':
            return self.Dice_loss_1by1
        elif mode == 'balanced_sig_focal_1by1':
            return self.balanced_sigmoid_focal_loss_1by1
        elif mode == 'sig_focal':
            return self.sigmoid_focal_loss
        elif mode == 'balanced_sig_focal':
            return self.balanced_sigmoid_focal_loss
        elif mode == 'balanced_focal_1by1':
            return self.balanced_focal_loss_1by1
        elif mode == 'dsc_loss':
            return self.cross_loss
        else:
            raise NotImplementedError

    def cross_loss(self, y_pred, y_true):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred + smooth) +
                           (1 - y_true) * torch.log(1 - y_pred + smooth))

    def Dice_loss_1by1(self, predict, target, epsilon=1.0):
        # predict = predict.sigmoid()
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        B, _, _, _ = predict.shape
        Dice_loss = []
        for idx in range(B):
            # pre_idx = predict[idx]
            # tar_idx = target[idx]
            # pre = pre_idx.view(1, -1)
            # tar = tar_idx.view(1, -1)
            # intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
            # union = (torch.square(pre) + torch.square(tar)).sum(-1).sum()
            # score = 1 - 2 * (intersection + epsilon) / (union + epsilon)
            pre_idx = predict[idx]
            tar_idx = target[idx]
            pre_idx = pre_idx.flatten(1)
            tar_idx = tar_idx.flatten(1)
            numerator = 2 * (pre_idx * tar_idx).sum(1)
            denominator = pre_idx.sum(-1) + tar_idx.sum(-1)
            score = 1 - (numerator + 1) / (denominator + 1)
            Dice_loss.append(score)

        batch_loss = torch.mean(torch.stack(Dice_loss).squeeze())

        return batch_loss  # mask:0.8, edge:0.9

    def balanced_sigmoid_focal_loss_1by1(self, inputs, targets, alpha: float = 0.25, gamma: float = 2):
        mask_balance = torch.ones_like(targets)
        if (targets == 1).sum():
            mask_balance[targets == 1] = 0.5 / ((targets == 1).sum().to(torch.float) / targets.numel())
            mask_balance[targets == 0] = 0.5 / ((targets == 0).sum().to(torch.float) / targets.numel())
        # else:
        # print('Mask balance is not working!')
        B, _, _, _ = inputs.shape
        Focal_loss = []
        for idx in range(B):
            input = inputs[idx].unsqueeze(0)
            target = targets[idx].unsqueeze(0)
            mask_balance_idx = mask_balance[idx].unsqueeze(0)
            prob = input.sigmoid()
            ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
            p_t = prob * target + (1 - prob) * (1 - target)
            loss = ce_loss * ((1 - p_t) ** gamma)
            if alpha >= 0:
                alpha_t = alpha * target + (1 - alpha) * (1 - target)
                loss = alpha_t * loss
            # n, c, h, w = input.size()
            loss = torch.mean(loss * mask_balance_idx)
            Focal_loss.append(loss)

        batch_loss = torch.mean(torch.stack(Focal_loss).squeeze())

        return batch_loss  # mask:0.09, edge:0.07

    def balanced_focal_loss_1by1(self, inputs, targets, alpha: float = 0.25, gamma: float = 2):
        mask_balance = torch.ones_like(targets)
        if (targets == 1).sum():
            mask_balance[targets == 1] = 0.5 / ((targets == 1).sum().to(torch.float) / targets.numel())
            mask_balance[targets == 0] = 0.5 / ((targets == 0).sum().to(torch.float) / targets.numel())
        # else:
        # print('Mask balance is not working!')
        B, _, _, _ = inputs.shape
        Focal_loss = []
        for idx in range(B):
            input = inputs[idx].unsqueeze(0)
            target = targets[idx].unsqueeze(0)
            mask_balance_idx = mask_balance[idx].unsqueeze(0)
            prob = input
            ce_loss = F.binary_cross_entropy(input, target, reduction="none")
            p_t = prob * target + (1 - prob) * (1 - target)
            loss = ce_loss * ((1 - p_t) ** gamma)
            if alpha >= 0:
                alpha_t = alpha * target + (1 - alpha) * (1 - target)
                loss = alpha_t * loss
            # n, c, h, w = input.size()
            loss = torch.mean(loss * mask_balance_idx)
            Focal_loss.append(loss)

        batch_loss = torch.mean(torch.stack(Focal_loss).squeeze())

        return batch_loss  # mask:0.09, edge:0.07

    def Dice_loss(self, predict, target, epsilon=1.0):
        # predict = predict.sigmoid()
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        # pre = torch.sigmoid(predict).view(num, -1)
        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        # union = (pre + tar).sum(-1).sum()
        # union2 = (pre + tar).sum(-1)
        # union3 = (torch.square(pre) + torch.square(tar))
        # union4 = union3.sum(-1)
        union = (torch.square(pre) + torch.square(tar)).sum(-1).sum()

        score = 1 - 2 * (intersection + epsilon) / (union + epsilon)

        return score  # mask:0.8, edge:0.9

    def sigmoid_focal_loss(self, inputs, targets, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        n, c, h, w = inputs.size()
        return loss.mean(1).sum()/h/w

    def balanced_sigmoid_focal_loss(self, inputs, targets, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        mask_balance = torch.ones_like(targets)
        if (targets == 1).sum():
            mask_balance[targets == 1] = 0.5 / ((targets == 1).sum().to(torch.float) / targets.numel())
            mask_balance[targets == 0] = 0.5 / ((targets == 0).sum().to(torch.float) / targets.numel())
        # else:
            # print('Mask balance is not working!')


        prob = inputs.sigmoid()
        # prob = inputs
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        n, c, h, w = inputs.size()
        loss = torch.mean(loss * mask_balance)
        return loss  # mask:0.09, edge:0.07


    def Focal_BCE(self, logit, target, gamma=2, alpha=0.25):
        n, c, h, w = logit.size()
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight,
                                        reduction='none')
        # criterion = nn.CrossEntropyLoss(weight=self.weight,
        #                                 reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        # print('logit', logit.shape)
        # print('target', target.shape)

        logpt = -criterion(logit, target)
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        # loss = -((1 - pt) ** gamma) * logpt
        loss1 = torch.pow(1.0 - pt, gamma)

        loss = -(loss1 * logpt)
        loss /= n
        loss = torch.mean(loss)
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='sum')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        loss /= n

        return loss

    def PPA_loss(self, pred, mask, kernel_size):
        """
        loss function (ref: F3Net-AAAI-2020)
        """
        # weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

        # weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        # wbce = F.binary_cross_entropy_with_logits(pred, mask, weight=self.weight, reduce='none')
        # wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        #
        # # pred = torch.sigmoid(pred)
        # inter = ((pred * mask) * weit).sum(dim=(2, 3))
        # union = ((pred + mask) * weit).sum(dim=(2, 3))
        # wiou = 1 - (inter + 1) / (union - inter + 1)
        # return (wbce + wiou).mean()

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def CELoss(self, pred, target):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12

        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log
            target = F.one_hot(target, self.class_num)  # 转换成one-hot

            # label smoothing
            # 实现 1
            # target = (1.0-self.label_smooth)*target + self.label_smooth/self.class_num
            # 实现 2
            # implement 2
            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1),
                                 max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * logprobs, 1)

        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))

        return loss.mean()

    # class MultiCEFocalLoss(torch.nn.Module):
    #     def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
    #         super(MultiCEFocalLoss, self).__init__()
    #         if alpha is None:
    #             self.alpha = Variable(torch.ones(class_num, 1))
    #         else:
    #             self.alpha = alpha
    #         self.gamma = gamma
    #         self.reduction = reduction
    #         self.class_num = class_num
    #
    #     def forward(self, predict, target):
    #         pt = F.softmax(predict, dim=1)  # softmmax获取预测概率
    #         class_mask = F.one_hot(target, self.class_num)  # 获取target的one hot编码
    #         ids = target.view(-1, 1)
    #         alpha = self.alpha[ids.data.view(-1)]  # 注意，这里的alpha是给定的一个list(tensor),里面的元素分别是每一个类的权重因子
    #         probs = (pt * class_mask).sum(1).view(-1, 1)  # 利用onehot作为mask，提取对应的pt
    #         log_p = probs.log()
    #         # 同样，原始ce上增加一个动态权重衰减因子
    #         loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
    #
    #         if self.reduction == 'mean':
    #             loss = loss.mean()
    #         elif self.reduction == 'sum':
    #             loss = loss.sum()
    #         return loss



if __name__ == "__main__":
    loss = SegmentationLosses()

    logit = torch.rand(8, 4, 7, 7)
    target = torch.rand(8, 7, 7)

   # a = a.view(8,196)

    print(loss.FocalLoss(logit, target, gamma=2, alpha=0.5).item())




