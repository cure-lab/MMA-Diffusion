# SD3.5 专用损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SD3CombinedLoss(nn.Module):
    """
    SD3.5 组合损失函数

    同时优化池化向量 (c_vec) 和上下文序列 (c_ctxt)

    L_total = λ1 * L_vec + λ2 * L_ctxt

    其中:
    - L_vec = 1 - cos(c_vec^adv, c_vec^tar)
    - L_ctxt = 1 - mean(cos(c_ctxt^adv_i, c_ctxt^tar_i))
    """

    def __init__(self, lambda_vec: float = 1.0, lambda_ctxt: float = 0.5):
        super().__init__()
        self.lambda_vec = lambda_vec
        self.lambda_ctxt = lambda_ctxt

    def forward(
        self,
        adv_pooled: torch.Tensor,          # [B, 2048] 对抗样本池化输出
        tar_pooled: torch.Tensor,          # [B, 2048] 目标池化输出
        adv_context: torch.Tensor = None,  # [B, 154, 4096] 对抗样本上下文 (可选)
        tar_context: torch.Tensor = None,  # [B, 154, 4096] 目标上下文 (可选)
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算组合损失"""
        # 1. 池化向量损失
        loss_vec = 1 - F.cosine_similarity(adv_pooled, tar_pooled, dim=-1).mean()

        # 如果没有上下文，只返回池化损失
        if adv_context is None or tar_context is None:
            return loss_vec

        # 2. 上下文序列损失
        B, T, D = adv_context.shape
        adv_flat = adv_context.reshape(-1, D)
        tar_flat = tar_context.reshape(-1, D)
        cos_sim = F.cosine_similarity(adv_flat, tar_flat, dim=-1)
        loss_ctxt = 1 - cos_sim.mean()

        return self.lambda_vec * loss_vec + self.lambda_ctxt * loss_ctxt


class SD3CombinedLoss_Batch(nn.Module):
    """
    SD3.5 批量损失函数（用于候选评估，返回每个样本的损失）
    """

    def __init__(self, lambda_vec: float = 1.0, lambda_ctxt: float = 0.5, reduction=None):
        super().__init__()
        self.lambda_vec = lambda_vec
        self.lambda_ctxt = lambda_ctxt
        self.reduction = reduction

    def forward(
        self,
        adv_pooled: torch.Tensor,          # [B, 2048]
        tar_pooled: torch.Tensor,          # [B, 2048] or [1, 2048]
        adv_context: torch.Tensor = None,  # [B, 154, 4096]
        tar_context: torch.Tensor = None,  # [B, 154, 4096] or [1, 154, 4096]
    ) -> torch.Tensor:
        """计算批量损失，返回 [B] 形状"""
        # 1. 池化向量损失
        loss_vec = 1 - F.cosine_similarity(adv_pooled, tar_pooled, dim=-1)  # [B]

        # 如果没有上下文，只返回池化损失
        if adv_context is None or tar_context is None:
            if self.reduction == 'mean':
                return loss_vec.mean()
            return loss_vec

        # 2. 上下文序列损失
        # 对每个 batch 计算序列维度的平均余弦相似度
        B = adv_context.shape[0]
        loss_ctxt_list = []
        for i in range(B):
            tar_idx = min(i, tar_context.shape[0] - 1)  # 处理 tar 可能只有 1 个的情况
            cos_sim = F.cosine_similarity(adv_context[i], tar_context[tar_idx], dim=-1)  # [154]
            loss_ctxt_list.append(1 - cos_sim.mean())
        loss_ctxt = torch.stack(loss_ctxt_list)  # [B]

        combined = self.lambda_vec * loss_vec + self.lambda_ctxt * loss_ctxt

        if self.reduction == 'mean':
            return combined.mean()
        return combined
