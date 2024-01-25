#!/usr/bin/env python3

from typing import Optional
from dataclasses import dataclass

import torch as th
import torch.nn as nn


def l1_color_t(pred_color: th.Tensor,
               true_color: th.Tensor,
               pred_mask: th.Tensor,
               true_mask: th.Tensor,
               learning_rates: th.Tensor,
               weight_color: float,) -> th.Tensor:
    # print(pred_color.shape)
    # print(true_color.shape)
    # print(pred_mask.shape)
    # print(true_mask.shape)
    diff_color = th.abs((pred_color * (pred_mask) - true_color) * (true_mask))
    batch_err = diff_color.reshape(diff_color.shape[0], -1).mean(dim=-1)
    lr_diff_color = batch_err * learning_rates
    return lr_diff_color.mean() * weight_color


def l1_depth_t(
        pred_depth: th.Tensor,
        true_depth: th.Tensor,
        true_mask: th.Tensor,
        learning_rates: th.Tensor,
        weight_depth: float,
        pred_mask: Optional[th.Tensor] = None):
    diff_depth = th.abs(
        (pred_depth - true_depth) * true_mask
    )
    batch_err = diff_depth.reshape(diff_depth.shape[0], -1).mean(dim=-1)
    lr_diff_depth = batch_err * learning_rates
    return lr_diff_depth.mean() * weight_depth


def l1_label_t(pred_mask: th.Tensor,
               true_mask: th.Tensor,
               learning_rates: th.Tensor,
               weight_mask: float):
    diff_mask = th.abs(pred_mask - true_mask)
    batch_err = diff_mask.reshape(diff_mask.shape[0], -1).mean(dim=-1)
    lr_diff_mask = batch_err * learning_rates
    return lr_diff_mask.mean() * weight_mask


class ImageLoss(nn.Module):

    @dataclass
    class Config:
        weight_color: float = 1.0
        weight_depth: float = 1.0
        weight_label: float = 1.0

    def __init__(self, cfg: Config,
                 learning_rates: th.Tensor):
        super().__init__()
        self.cfg = cfg
        self.learning_rates = learning_rates

    def forward(self,
                pred_color: Optional[th.Tensor],
                pred_depth: Optional[th.Tensor],
                pred_label: Optional[th.Tensor],
                pred_valid: Optional[th.Tensor],
                true_color: th.Tensor,
                true_label: th.Tensor):
        cfg = self.cfg
        # FIXME(ycho): depth/label losses currenly omitted
        return l1_color_t(pred_color,
                          true_color,
                          # pred_label,
                          pred_valid,
                          true_label,
                          self.learning_rates,
                          cfg.weight_color)
