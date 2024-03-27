import math
from typing import Dict

import numpy as np
import torch
from torch import nn
import logging

log = logging.getLogger()


def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec


class DRLoss:
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 reg_lambda=0.
                 ):
        super().__init__()

        self.reduction = reduction  # unused, mean used by default rather than alternatives such as sum
        self.loss_weight = loss_weight  # 10. in usage by original paper
        self.reg_lambda = reg_lambda  # unused

    def calc(
            self,
            feat,
            target,
            h_norm2=None,
            m_norm2=None,
            avg_factor=None,
    ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)

        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)

        return loss * self.loss_weight


class ETFHead(nn.Module):
    """Classification head for Baseline.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
    """

    def __init__(self, num_classes: int, in_channels: int, *args, **kwargs) -> None:
        if kwargs.get('eval_classes', None):
            self.eval_classes = kwargs.pop('eval_classes')
        else:
            self.eval_classes = num_classes

        # training settings about different length for different classes
        if kwargs.pop('with_len', False):
            self.with_len = True
        else:
            self.with_len = False

        super().__init__(*args, **kwargs)
        assert num_classes > 0, f'num_classes={num_classes} must be a positive integer'

        self.num_classes = num_classes
        self.in_channels = in_channels
        log.info("ETF head : evaluating {} out of {} classes.".format(self.eval_classes, self.num_classes))
        log.info("ETF head : with_len : {}".format(self.with_len))

        orth_vec = generate_random_orthogonal_matrix(self.in_channels, self.num_classes)
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(self.num_classes, self.num_classes), (1 / self.num_classes))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(self.num_classes / (self.num_classes - 1)))
        self.register_buffer('etf_vec', etf_vec)

        etf_rect = torch.ones((1, num_classes), dtype=torch.float32)
        self.etf_rect = etf_rect

        self.loss_calculator = DRLoss(loss_weight=10.)

    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

    def forward_train(self, x: torch.Tensor, gt_label: torch.Tensor, **kwargs) -> Dict:
        """Forward training data."""
        x = self.pre_logits(x)
        # Equivalent to performing a weighted loss function on different classes, according to Appendix C of Yang et al, 2022
        if self.with_len:
            etf_vec = self.etf_vec * self.etf_rect.to(device=self.etf_vec.device)
            target = (etf_vec * self.produce_training_rect(gt_label, self.num_classes))[:, gt_label].t()
        else:
            target = self.etf_vec[:, gt_label].t()
        losses = self.loss(x, target)
        if self.cal_acc:
            with torch.no_grad():
                cls_score = x @ self.etf_vec
                acc = self.compute_accuracy(cls_score[:, :self.eval_classes], gt_label)
                assert len(acc) == len(self.topk)
                losses['accuracy'] = {
                    f'top-{k}': a
                    for k, a in zip(self.topk, acc)
                }
        return losses

    def mixup_extra_training(self, x: torch.Tensor) -> Dict:
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        assigned = torch.argmax(cls_score[:, self.eval_classes:], dim=1)
        target = self.etf_vec[:, assigned + self.eval_classes].t()
        losses = self.loss(x, target)
        return losses

    def loss(self, feat, target, **kwargs) -> Dict:
        losses = dict()
        # No longer using mmcv
        # compute loss
        if self.with_len:
            loss = self.loss_calculator.calc(feat, target, m_norm2=torch.norm(target, p=2, dim=1))
        else:
            loss = self.loss_calculator.calc(feat, target)
        losses['loss'] = loss
        return losses

    def simple_test(self, x, softmax=False, post_process=False):
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        cls_score = cls_score[:, :self.eval_classes]
        assert not softmax
        if post_process:
            return self.post_process(cls_score)
        else:
            return cls_score

    @staticmethod
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
