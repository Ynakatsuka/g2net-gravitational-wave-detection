import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(
        self, gamma=0, alpha=None, size_average=True, ignore_index=-100
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        selfignore_index = ignore_index

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(
                input.size(0), input.size(1), -1
            )  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(
                -1, input.size(2)
            )  # N,H*W,C => N*H*W,C

        target = target * (target != self.ignore_index).float()
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class BinaryFocalLoss(nn.Module):
    def __init__(
        self, gamma=2, alpha=None, pos_weight=None, ignore_index=-100, **_
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, input, target, reduction=True, weight=None):
        target = target * (target != self.ignore_index).float()

        input = input.view(-1, 1)
        target = target.view(-1, 1)
        assert (
            target.size() == input.size()
        ), f"{target.size()} vs {input.size()}"
        if weight is not None:
            assert target.size() == weight.size()

        # For test
        if isinstance(self.pos_weight, float) or isinstance(
            self.pos_weight, int
        ):
            weight = target * (self.pos_weight - 1.0) + 1.0
        else:
            if weight is None:
                weight = target + 2.0

        max_val = (-input).clamp(min=0)

        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        if weight is not None:
            loss = loss * weight

        if reduction:
            return loss.mean()
        else:
            return loss


class BinaryReducedFocalLoss(nn.Module):
    def __init__(
        self, gamma=2, alpha=None, pos_weight=None, threshold=0.5, **_
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.threshold = threshold

    def forward(self, input, target, reduction=True, weight=None):
        target = target.float()

        input = input.view(-1, 1)
        target = target.view(-1, 1)
        assert (
            target.size() == input.size()
        ), f"{target.size()} vs {input.size()}"
        if weight is not None:
            assert target.size() == weight.size()

        # For test
        if isinstance(self.pos_weight, float) or isinstance(
            self.pos_weight, int
        ):
            weight = target * (self.pos_weight - 1.0) + 1.0
        else:
            if weight is None:
                weight = target + 2.0

        max_val = (-input).clamp(min=0)

        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        invprobs = torch.where(
            input.gt(0),
            invprobs,
            torch.tensor(0.0).float().to(invprobs.device),
        )

        loss = (invprobs / self.threshold * self.gamma).exp() * loss
        if weight is not None:
            loss = loss * weight

        if reduction:
            return loss.mean()
        else:
            return loss


class LabelSmoothBinaryFocalLoss(nn.Module):
    def __init__(
        self, lb_smooth=0.1, gamma=2, alpha=None, pos_weight=None, **_
    ):
        super().__init__()
        self.lb_smooth = lb_smooth
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, input, target, reduction=True, weight=None):
        target = target.float()
        target = target * (1 - self.lb_smooth) + self.lb_smooth / target.size(
            1
        )

        input = input.view(-1, 1)
        target = target.view(-1, 1)
        assert (
            target.size() == input.size()
        ), f"{target.size()} vs {input.size()}"
        if weight is not None:
            assert target.size() == weight.size()

        # For test
        if isinstance(self.pos_weight, float) or isinstance(
            self.pos_weight, int
        ):
            weight = target * (self.pos_weight - 1.0) + 1.0
        else:
            if weight is None:
                weight = target + 2.0

        max_val = (-input).clamp(min=0)

        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        if weight is not None:
            loss = loss * weight

        if reduction:
            return loss.mean()
        else:
            return loss


class BinaryDualFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, pos_weight=None, **_):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, input, target, reduction=True, weight=None):
        target = target.float()

        input = input.view(-1, 1)
        target = target.view(-1, 1)
        assert (
            target.size() == input.size()
        ), f"{target.size()} vs {input.size()}"
        if weight is not None:
            assert target.size() == weight.size()

        # For test
        if isinstance(self.pos_weight, float) or isinstance(
            self.pos_weight, int
        ):
            weight = target * (self.pos_weight - 1.0) + 1.0
        else:
            if weight is None:
                weight = target + 2.0

        proba = F.sigmoid(input)
        diff = torch.abs(target - proba)
        loss = -(diff * (torch.log(1.0 - diff) ** self.gamma)).sum(dim=1)

        if weight is not None:
            loss = loss * weight

        if reduction:
            return loss.mean()
        else:
            return loss
