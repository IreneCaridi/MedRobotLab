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
        targets = targets.view(-1)
        loss = 0.0

        for c in range(inputs.size(1)):
            # Get the mask for the current class
            mask = (targets == c)
            # Calculate the focal loss for this class
            p_t = inputs[:, c]
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