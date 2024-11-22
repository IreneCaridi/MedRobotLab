import torch
import torch.nn as nn
import torch.nn.functional as F


# look at it, simply copied from chatgpt
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Apply softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)
        targets = targets
        loss = 0.0

        for c in range(inputs.size(1)):
            # Get the mask for the current class
            mask = (targets == c)
            # Calculate the focal loss for this class
            p_t = inputs[:, c, :, :]
            loss -= self.alpha * mask * (1 - p_t) ** self.gamma * torch.log(p_t)

        return loss.mean()


class SmoothL2Loss(nn.Module):  # hubert loss
    def __init__(self, delta=1.0):
        super(SmoothL2Loss, self).__init__()
        self.delta = delta

    def forward(self, y_true, y_pred):
        diff = torch.abs(y_true - y_pred)
        loss = torch.where(diff < self.delta,
                           0.5 * diff**2,
                           self.delta * (diff - 0.5 * self.delta))
        return loss.mean()


def _reshape_mask(mask):
    return mask.reshape(mask.shape[0] * mask.shape[1], mask.shape[2] * mask.shape[3])


def dice_loss(inputs, targets, valid=None, target_logit=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = _reshape_mask(inputs)

    if target_logit:
        targets = targets.sigmoid()
    targets = _reshape_mask(targets)

    if valid is not None:
        valid = _reshape_mask(valid)
        inputs = inputs * valid
        targets = targets * valid

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


class SemanticFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        """
            Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            - alpha: Weighting factor in range (0,1). (I think is useless for multiclass)
            - gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
            - weight: torch.Tensor of weights per class (expected normalized)
        """

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

        self.weight = weight


    def forward(self, x, y):
        """
        Args:
            x: Logits as output of model with shape BxCxHxW.
            y: Masks tensor as integers with shae BxHxW.
        Returns:
            Loss: loss averaged
        """

        ce_loss = F.cross_entropy(x, y, reduction='none', weight=self.weight)  # shape BxHxW
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # shape BxHxW

        return loss.mean()


class SemanticDiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-6):
        """
            Dice Loss for Semantic Segmentation.

        Args:
            weight: torch.Tensor of weights per class (expected normalized)
            smooth: (float, optional): Smoothing term to avoid division by zero.
        """
        super().__init__()

        self.smooth = smooth
        self.weight = weight  # shape C

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Predicted logits of shape [N, C, H, W].
            y (torch.Tensor): Ground truth labels of shape [N, H, W].

        Returns:
            torch.Tensor: Dice loss.
        """

        N, C = x.size()[:2]
        # Flatten spatial dimensions
        x = F.softmax(x, dim=1).view(N, C, -1)  # [N, C, *]

        y = y.view(N, -1)  # [N, *]

        # One-hot encoding for the target
        y_onehot = F.one_hot(y, num_classes=C).permute(0, 2, 1)  # [N, C, *]

        # Compute intersection and union
        intersection = torch.sum(x * y_onehot, dim=2)  # [N, C]
        union = torch.sum(x.pow(2), dim=2) + torch.sum(y_onehot, dim=2)  # [N, C]

        # Compute Dice coefficient
        dice = (2 * intersection + self.smooth) / (union + self.smooth)  # [N, C]

        # Apply class weights if available
        if self.weight:
            weight = self.weight.to(x.device)
            dice = dice * weight  # [N, C]

        # Average over classes and batches
        loss = 1 - torch.mean(dice)

        return loss


class SemanticLosses(nn.Module):
    def __init__(self, alpha=1, gamma=2, lambdas: tuple = (0.5, 0.5), weight=None):
        """
            handler for the losses to be used Semantic Segmentation scenario

        args:
            - alpha: focal loss alpha parameter (I think is useless for multiclass usage, use weights instead)
            - gamma: focal loss gamma parameter
            - lambdas: tuple with relative losses weights (1st is focal, 2nd Dice)
            - weight: list of class weights of len == N (num_classes)
        """
        super().__init__()

        if weight:
            weight = torch.tensor(weight, dtype=torch.float32)
            weight = weight / torch.sum(weight)   # ensures normalization

        self.loss1 = SemanticFocalLoss(alpha, gamma, weight)
        self.loss2 = SemanticDiceLoss(weight)

        self.lambda1, self.lambda2 = lambdas

    def forward(self, x, y):
        focal = self.loss1(x, y)
        dice = self.loss2(x, y)

        return self.lambda1 * focal + self.lambda2 * dice







