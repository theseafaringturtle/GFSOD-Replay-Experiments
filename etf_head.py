import math

import numpy as np
import torch
from torch import nn, Tensor
import logging

# This code was adapted from the implementation by Yang et al, 2023 https://github.com/NeuralCollapseApplications/FSCIL which used OpenMMLab

log = logging.getLogger()


def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec


def produce_training_rectifier(label: torch.Tensor, num_classes: int):
    """Scales classifier weights based on class imbalance across a batch. Yang et al, 2022, Appendix C"""
    # Get array of how many times each label appears
    uni_label, count = torch.unique(label, return_counts=True)
    batch_size = label.size(0)
    uni_label_num = uni_label.size(0)
    assert batch_size == torch.sum(count)
    # Return array of ratios
    gamma = torch.tensor(batch_size / uni_label_num, device=label.device, dtype=torch.float32)
    rect = torch.ones(1, num_classes).to(device=label.device, dtype=torch.float32)
    rect[0, uni_label] = torch.sqrt(gamma / count)
    return rect


def l2_norm_unit(x):
    """Convert vectors to l2 norms scaled 0-1"""
    x = x / torch.norm(x, p=2, dim=1, keepdim=True)
    return x


class ETFHead(nn.Module):
    """Classification head for Baseline.
    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
    """

    def __init__(self, num_classes: int, in_channels: int) -> None:
        super().__init__()
        self.eval_classes = num_classes

        # (original comment) training settings about different length for different classes
        assert num_classes > 0, f'num_classes={num_classes} must be a positive integer'

        self.num_classes = num_classes
        self.in_channels = in_channels
        log.info("ETF head : evaluating {} out of {} classes.".format(self.eval_classes, self.num_classes))

        orth_vec = generate_random_orthogonal_matrix(self.in_channels, self.num_classes)
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(self.num_classes, self.num_classes), (1 / self.num_classes))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(self.num_classes / (self.num_classes - 1)))
        self.register_buffer('etf_vec', etf_vec)

        etf_rect = torch.ones((1, num_classes), dtype=torch.float32)
        self.etf_rect = etf_rect

    def predict_logits(self, x):
        with torch.no_grad():
            x = l2_norm_unit(x)
            cls_score = x @ self.etf_vec
            cls_score = cls_score[:, :self.eval_classes]  # only useful if option for eval_classes != num_classes
            return cls_score


class DRLoss(nn.Module):
    def __init__(self, etf_head: ETFHead, rectify_imbalance=False, loss_weight=10.0):
        super().__init__()

        self.rectify_imbalance = rectify_imbalance  # same as with_len in original impl
        self.loss_weight = loss_weight  # 10. in usage by original paper
        self.etf_head = etf_head
        # self.reduction = reduction  # unused, mean used by default rather than alternatives such as sum
        # self.reg_lambda = reg_lambda  # unused

    def forward(self, outputs, targets):
        """Loss function"""
        x = l2_norm_unit(outputs)
        if self.rectify_imbalance:
            etf_vec = self.etf_head.etf_vec * self.etf_head.etf_rect  # .to(device=self.etf_head.etf_vec.device)
            target = (etf_vec * produce_training_rectifier(targets, self.etf_head.num_classes))[:, targets].t()
            return self.calc(x, target, m_norm2=torch.norm(target, p=2, dim=1))
        else:
            target = self.etf_head.etf_vec[:, targets].t()
            return self.calc(x, target)

    def calc(self, feat, target, m_norm2=None):
        """Dot-regression loss"""
        dot = torch.sum(feat * target, dim=1, )
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)
        loss = 0.5 * torch.mean((dot - m_norm2) ** 2)
        return loss * self.loss_weight
