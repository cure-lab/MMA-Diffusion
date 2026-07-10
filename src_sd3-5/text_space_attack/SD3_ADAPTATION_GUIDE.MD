# SD3.5 文本空间攻击适配指南

> 本文档详细说明如何将 SD1.5 的 MMA（Masked Multimodal Attack）文本攻击方法适配到 SD3.5 模型。

---

## 0. 复现实验快速检查

1. 准备独立 Python/conda 环境，并安装原仓库环境依赖。SD3.5 入口依赖 `diffusers`、`transformers`、`datasets`、`accelerate`、`safetensors`、`sentencepiece` 和 `torch`；当前 encoder patch 已在 `transformers==4.57.3` 上验证。
2. 应用 CLIP encoder 补丁。SD3.5 的梯度注入会调用 `input_embed`，标准 `transformers` CLIP 不支持该参数。可以在环境的 `site-packages` 目录执行：
   ```bash
   cd <python-env>/lib/python3.10/site-packages
   patch -p0 < /path/to/MMA-Diffusion/patches/transformers_clip_input_embed.patch
   ```
   如果你的 `transformers` 版本不同，按第 5 节手动修改 `transformers/models/clip/modeling_clip.py`。
3. 确认可访问 `stabilityai/stable-diffusion-3.5-medium`。如果 Hugging Face 要求授权，请先接受模型许可并登录；也可以用 `--model_name /path/to/local/sd3.5` 指向本地模型目录。
4. 运行攻击：
   ```bash
   cd src_sd3-5
   python textual_attack.py --limit 16 --batch_size 4 --n_steps 500 --disable_swanlab
   ```
   默认数据集为 `YijunYang280/MMA-Diffusion-NSFW-adv-prompts-benchmark`，也可以用 `--data_file path/to/prompts.csv --prompt_column target_prompt` 指定本地 CSV。
5. 评估输出：
   ```bash
   python evaluate_text_sd3.5.py --csv_path stable-diffusion-3.5-medium/no_safe_checker/attack_results.csv
   ```

---

## 1. 背景：SD1.5 vs SD3.5 架构差异

### 1.1 文本编码器对比

| 特性 | SD1.5 | SD3.5 |
|------|-------|-------|
| 文本编码器数量 | 1 个 | 3 个 |
| 编码器类型 | CLIP-ViT/L | CLIP-L/14 + OpenCLIP-G/14 + T5-XXL |
| 池化输出维度 | 768 维 | 2048 维 (768+1280) |
| 序列输出维度 | 77×768 | 154×4096 |
| 条件注入方式 | Cross-Attention | MM-DiT 双流架构 |

### 1.2 SD3.5 的三编码器架构

SD3.5 使用 `StableDiffusion3Pipeline`，包含三个文本编码器：

```python
# 访问方式
pipeline.text_encoder      # CLIP-L/14 (768 dim)
pipeline.text_encoder_2    # OpenCLIP-G/14 (1280 dim)
pipeline.text_encoder_3    # T5-XXL (4096 dim)

# 对应的 tokenizer
pipeline.tokenizer         # CLIP tokenizer
pipeline.tokenizer_2       # CLIP tokenizer
pipeline.tokenizer_3       # T5 tokenizer
```

---

## 2. SD3.5 文本编码流程详解

### 2.1 完整编码流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SD3.5 文本编码架构                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: "a photo of a cat"                                                   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   CLIP-L/14      │  │  OpenCLIP-G/14   │  │     T5-XXL       │          │
│  │   Tokenizer      │  │    Tokenizer     │  │    Tokenizer     │          │
│  │   (vocab ~49k)   │  │   (vocab ~49k)   │  │   (vocab ~32k)   │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                     │                     │
│           ▼                     ▼                     ▼                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   CLIP-L/14      │  │  OpenCLIP-G/14   │  │     T5-XXL       │          │
│  │   Encoder        │  │    Encoder       │  │    Encoder       │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                     │                     │
│     ┌─────┴─────┐         ┌─────┴─────┐               │                     │
│     ▼           ▼         ▼           ▼               ▼                     │
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   ┌────────────────┐           │
│ │ pooled │ │hidden  │ │ pooled │ │hidden  │   │ last_hidden    │           │
│ │  768d  │ │77×768  │ │ 1280d  │ │77×1280 │   │    77×4096     │           │
│ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘   └───────┬────────┘           │
│     │          │          │          │                │                     │
│     │          └────┬─────┴──────────┘                │                     │
│     │               │                                 │                     │
│     │               ▼                                 │                     │
│     │      ┌─────────────────┐                        │                     │
│     │      │ 拼接隐藏层       │                        │                     │
│     │      │ 77×(768+1280)   │                        │                     │
│     │      │ = 77×2048       │                        │                     │
│     │      └────────┬────────┘                        │                     │
│     │               │                                 │                     │
│     │               ▼                                 │                     │
│     │      ┌─────────────────┐                        │                     │
│     │      │ 补零至 4096 维   │                        │                     │
│     │      │ 77×4096         │                        │                     │
│     │      └────────┬────────┘                        │                     │
│     │               │                                 │                     │
│     │               └──────────┬──────────────────────┘                     │
│     │                          │                                            │
│     │                          ▼                                            │
│     │                 ┌─────────────────┐                                   │
│     │                 │ Token维度拼接   │                                   │
│     │                 │ c_ctxt:         │                                   │
│     │                 │ (77+77)×4096    │                                   │
│     │                 │ = 154×4096      │                                   │
│     │                 └────────┬────────┘                                   │
│     │                          │                                            │
│     ▼                          ▼                                            │
│ ┌────────────────┐    ┌────────────────────┐                                │
│ │ 池化输出拼接   │    │ 上下文嵌入          │                                │
│ │ c_vec:         │    │ 用于 MM-DiT        │                                │
│ │ 768+1280=2048d │    │ 文本流输入          │                                │
│ │                │    │                    │                                │
│ │ 用于时间步调制 │    │                    │                                │
│ └────────────────┘    └────────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 输出说明

| 输出 | 维度 | 用途 | 计算方式 |
|------|------|------|----------|
| `c_vec` (池化向量) | 2048 | 与时间步嵌入结合，用于调制 | `concat(clip_l_pooled, clip_g_pooled)` |
| `c_ctxt` (上下文序列) | 154×4096 | 作为 MM-DiT 的文本输入流 | 见下方详细说明 |

**`c_ctxt` 的构建步骤：**
1. 提取 CLIP-L 的倒数第二层隐藏状态: `77×768`
2. 提取 OpenCLIP-G 的倒数第二层隐藏状态: `77×1280`
3. 拼接两个 CLIP 隐藏状态: `77×2048`
4. 在通道维度补零至 4096 维: `77×4096`
5. 提取 T5-XXL 的最后一层隐藏状态: `77×4096`
6. 在 token 维度拼接: `(77+77)×4096 = 154×4096`

---

## 3. 损失函数设计

### 3.1 SD1.5 原始损失

SD1.5 仅使用单一 CLIP 编码器的池化输出：

$$\mathcal{L}_{SD1.5} = 1 - \cos(c^{adv}, c^{tar})$$

其中 $c^{adv}$ 是对抗样本的嵌入，$c^{tar}$ 是目标（恶意）提示的嵌入。

### 3.2 SD3.5 改进损失

对于 SD3.5，需要同时优化两个关键输出以确保攻击效果：

$$\boxed{\mathcal{L}_{total} = \lambda_1 \cdot \text{sim}(c_{vec}^{adv}, c_{vec}^{tar}) + \lambda_2 \cdot \text{sim}(c_{ctxt}^{adv}, c_{ctxt}^{tar})}$$

展开为：

$$\mathcal{L}_{total} = \lambda_1 \cdot \mathcal{L}_{vec} + \lambda_2 \cdot \mathcal{L}_{ctxt}$$

其中：

- **池化向量损失**（影响时间步调制）：
$$\mathcal{L}_{vec} = 1 - \cos(c_{vec}^{adv}, c_{vec}^{tar})$$

- **上下文序列损失**（影响 MM-DiT 文本流）：
$$\mathcal{L}_{ctxt} = 1 - \frac{1}{N}\sum_{i=1}^{N}\cos(c_{ctxt,i}^{adv}, c_{ctxt,i}^{tar})$$

其中 $N=154$ 是序列长度。

### 3.3 参数建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| $\lambda_1$ | 1.0 | 池化向量权重（影响整体语义） |
| $\lambda_2$ | 0.5 | 上下文序列权重（影响细节表示） |

> **注意**：可以根据实验效果调整这两个权重。如果发现攻击效果不佳，可以尝试增大 $\lambda_2$。

---

## 4. 代码修改详解

### 4.1 文件结构

```
src_sd3-5/
├── textual_attack.py              # 主入口（需修改）
└── text_space_attack/
    ├── SD3_ADAPTATION_GUIDE.md    # 本文档
    ├── base.py                    # 基类（基本不变）
    ├── MMA_para.py                # 核心攻击逻辑（需大改）
    ├── model.py                   # 模型加载（需修改）
    └── sd3_loss.py                # 【新增】SD3专用损失函数
```

### 4.2 model.py 修改

将原来的 `StableDiffusionPipeline` 替换为 `StableDiffusion3Pipeline`：

```python
# filepath: src_sd3-5/text_space_attack/model.py
from diffusers import StableDiffusion3Pipeline
import torch

MODEL_NAME = {
    "stable-diffusion-3.5-medium": "stabilityai/stable-diffusion-3.5-medium",
}

def get_model_path(model_name):
    return MODEL_NAME.get(model_name, model_name)


class GenerativeModel:
    """SD3.5 生成模型包装器"""

    def __init__(
        self,
        model_path: str = "stable-diffusion-3.5-medium",
        device: str = "cuda",
        torch_dtype=torch.float16,
        use_safe_checker: bool = False  # SD3.5 默认没有 safety_checker
    ):
        model_path = get_model_path(model_path)

        self.model = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            use_safetensors=True
        ).to(device)

        self.device = device
        self.model_path = model_path

    def generate(self, prompt: str, **kwargs):
        """生成图像"""
        return self.model(
            prompt,
            num_inference_steps=kwargs.get('num_inference_steps', 28),
            guidance_scale=kwargs.get('guidance_scale', 7.0),
        )
```

### 4.3 新增 sd3_loss.py

```python
# filepath: src_sd3-5/text_space_attack/sd3_loss.py
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
        adv_context: torch.Tensor,         # [B, 154, 4096] 对抗样本上下文
        tar_context: torch.Tensor,         # [B, 154, 4096] 目标上下文
        attention_mask: Optional[torch.Tensor] = None  # [B, 154] 可选的注意力掩码
    ) -> torch.Tensor:
        """
        计算组合损失

        Args:
            adv_pooled: 对抗样本的池化输出 [B, 2048]
            tar_pooled: 目标的池化输出 [B, 2048]
            adv_context: 对抗样本的上下文序列 [B, 154, 4096]
            tar_context: 目标的上下文序列 [B, 154, 4096]
            attention_mask: 可选的注意力掩码 [B, 154]

        Returns:
            组合损失值
        """
        # 1. 池化向量损失
        loss_vec = 1 - F.cosine_similarity(adv_pooled, tar_pooled, dim=-1).mean()

        # 2. 上下文序列损失
        if attention_mask is not None:
            # 归一化
            adv_norm = F.normalize(adv_context, dim=-1)
            tar_norm = F.normalize(tar_context, dim=-1)

            # 计算每个位置的余弦相似度
            cos_sim = (adv_norm * tar_norm).sum(dim=-1)  # [B, 154]

            # 应用掩码并计算平均
            masked_sim = cos_sim * attention_mask
            loss_ctxt = 1 - (masked_sim.sum(dim=-1) / (attention_mask.sum(dim=-1) + 1e-8)).mean()
        else:
            # 无掩码：简单展平并计算平均余弦相似度
            B, T, D = adv_context.shape
            adv_flat = adv_context.reshape(-1, D)
            tar_flat = tar_context.reshape(-1, D)
            cos_sim = F.cosine_similarity(adv_flat, tar_flat, dim=-1)
            loss_ctxt = 1 - cos_sim.mean()

        return self.lambda_vec * loss_vec + self.lambda_ctxt * loss_ctxt


class SD3PooledOnlyLoss(nn.Module):
    """
    仅使用池化输出的简化损失（可用于快速实验）

    这个损失函数仅优化 c_vec，忽略 c_ctxt。
    适用于：
    - 快速原型测试
    - 显存受限的情况
    - 对比实验
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        adv_pooled: torch.Tensor,
        tar_pooled: torch.Tensor
    ) -> torch.Tensor:
        """
        计算池化向量损失

        Args:
            adv_pooled: 对抗样本的池化输出 [B, 2048]
            tar_pooled: 目标的池化输出 [B, 2048]

        Returns:
            损失值
        """
        return 1 - F.cosine_similarity(adv_pooled, tar_pooled, dim=-1).mean()


class CosineSimilarityLoss(nn.Module):
    """
    原始的余弦相似度损失（与 SD1.5 兼容）
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos_sim = F.cosine_similarity(x, y, dim=1, eps=1e-6)
        loss = 1 - cos_sim
        return loss.mean()
```

### 4.4 MMA_para.py 核心修改

#### 4.4.1 初始化部分 - 支持三个编码器

```python
# filepath: src_sd3-5/text_space_attack/MMA_para.py
# 修改 __init__ 方法

class ParallelMMA(BaseAttacker):
    """
    SD3.5 版本的 Parallel MMA 攻击
    """

    def __init__(
        self,
        target_model: Any = None,
        text_detector: Optional[Callable[[str], bool]] = None,
        image_detector: Optional[Callable[[Any], bool]] = None,
        save_dir: str = None,
        **kwargs
    ):
        super().__init__(target_model, text_detector, image_detector)

        # ========== SD3.5 有三个文本编码器 ==========
        self.text_encoder_1 = self.target_model.model.text_encoder      # CLIP-L/14
        self.text_encoder_2 = self.target_model.model.text_encoder_2    # OpenCLIP-G/14
        self.text_encoder_3 = self.target_model.model.text_encoder_3    # T5-XXL

        # 对应的 tokenizer
        self.tokenizer_1 = self.target_model.model.tokenizer            # CLIP tokenizer
        self.tokenizer_2 = self.target_model.model.tokenizer_2          # CLIP tokenizer
        self.tokenizer_3 = self.target_model.model.tokenizer_3          # T5 tokenizer

        # 向后兼容：保留原始属性名
        self.text_encoder = self.text_encoder_1
        self.tokenizer = self.tokenizer_1

        # 设置为评估模式并冻结参数
        for encoder in [self.text_encoder_1, self.text_encoder_2, self.text_encoder_3]:
            if encoder is not None:
                encoder.eval()
                for p in encoder.parameters():
                    p.requires_grad_(False)

        self.device = next(self.text_encoder_1.parameters()).device

        # ========== 损失函数参数 ==========
        self.lambda_vec = kwargs.get('lambda_vec', 1.0)
        self.lambda_ctxt = kwargs.get('lambda_ctxt', 0.5)

        from .sd3_loss import SD3CombinedLoss
        self.loss_fn = SD3CombinedLoss(self.lambda_vec, self.lambda_ctxt)

        # ========== 其他参数保持不变 ==========
        self.threshold = kwargs.get("threshold", 0.1)
        self.internal_batch_size = kwargs.get("internal_batch_size", 512)
        self.n_steps = kwargs.get("n_steps", 500)
        self.topk = kwargs.get("topk", 256)
        self.control_length = kwargs.get("control_length", 20)
        self.early_stopping = kwargs.get("early_stopping", True)
        # ... 其他参数 ...
```

#### 4.4.2 新增编码函数 - 获取完整的 SD3 表示

```python
# filepath: src_sd3-5/text_space_attack/MMA_para.py
# 新增方法

class ParallelMMA(BaseAttacker):
    # ... __init__ ...

    def encode_prompt_sd3(
        self,
        prompt: str,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        对单个 prompt 进行完整的 SD3.5 编码

        Args:
            prompt: 输入文本
            return_dict: 是否返回字典

        Returns:
            dict with keys:
                - pooled_output: [2048] 池化向量 (c_vec)
                - context_output: [154, 4096] 上下文序列 (c_ctxt)
                - clip_l_hidden: [77, 768] CLIP-L 隐藏层
                - clip_g_hidden: [77, 1280] OpenCLIP-G 隐藏层
                - t5_hidden: [77, 4096] T5 隐藏层
        """
        with torch.no_grad():
            # 1. CLIP-L/14 编码
            tokens_1 = self.tokenizer_1(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs_1 = self.text_encoder_1(
                tokens_1.input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            clip_l_hidden = outputs_1.hidden_states[-2]  # 倒数第二层 [1, 77, 768]
            clip_l_pooled = outputs_1.pooler_output      # [1, 768]

            # 2. OpenCLIP-G/14 编码
            tokens_2 = self.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs_2 = self.text_encoder_2(
                tokens_2.input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            clip_g_hidden = outputs_2.hidden_states[-2]  # 倒数第二层 [1, 77, 1280]
            clip_g_pooled = outputs_2.pooler_output      # [1, 1280]

            # 3. T5-XXL 编码
            tokens_3 = self.tokenizer_3(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs_3 = self.text_encoder_3(
                tokens_3.input_ids,
                return_dict=True
            )
            t5_hidden = outputs_3.last_hidden_state  # [1, 77, 4096]

            # 4. 构建池化输出 c_vec (2048 维)
            pooled_output = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)  # [1, 2048]

            # 5. 构建上下文序列 c_ctxt (154 × 4096)
            # CLIP 拼接
            clip_combined = torch.cat([clip_l_hidden, clip_g_hidden], dim=-1)  # [1, 77, 2048]
            # 补零至 4096 维
            clip_padded = F.pad(clip_combined, (0, 2048))  # [1, 77, 4096]
            # 与 T5 在 token 维度拼接
            context_output = torch.cat([clip_padded, t5_hidden], dim=1)  # [1, 154, 4096]

        if return_dict:
            return {
                'pooled_output': pooled_output.squeeze(0),    # [2048]
                'context_output': context_output.squeeze(0),  # [154, 4096]
                'clip_l_hidden': clip_l_hidden.squeeze(0),    # [77, 768]
                'clip_g_hidden': clip_g_hidden.squeeze(0),    # [77, 1280]
                't5_hidden': t5_hidden.squeeze(0)             # [77, 4096]
            }
        return pooled_output.squeeze(0), context_output.squeeze(0)

    def encode_prompt_sd3_batch(
        self,
        prompts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        批量编码多个 prompt

        Args:
            prompts: 输入文本列表

        Returns:
            dict with keys:
                - pooled_output: [B, 2048]
                - context_output: [B, 154, 4096]
        """
        batch_size = len(prompts)

        with torch.no_grad():
            # 1. CLIP-L/14
            tokens_1 = self.tokenizer_1(
                prompts,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs_1 = self.text_encoder_1(
                tokens_1.input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            clip_l_hidden = outputs_1.hidden_states[-2]
            clip_l_pooled = outputs_1.pooler_output

            # 2. OpenCLIP-G/14
            tokens_2 = self.tokenizer_2(
                prompts,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs_2 = self.text_encoder_2(
                tokens_2.input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            clip_g_hidden = outputs_2.hidden_states[-2]
            clip_g_pooled = outputs_2.pooler_output

            # 3. T5-XXL
            tokens_3 = self.tokenizer_3(
                prompts,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs_3 = self.text_encoder_3(
                tokens_3.input_ids,
                return_dict=True
            )
            t5_hidden = outputs_3.last_hidden_state

            # 构建输出
            pooled_output = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)  # [B, 2048]
            clip_combined = torch.cat([clip_l_hidden, clip_g_hidden], dim=-1)  # [B, 77, 2048]
            clip_padded = F.pad(clip_combined, (0, 2048))                       # [B, 77, 4096]
            context_output = torch.cat([clip_padded, t5_hidden], dim=1)         # [B, 154, 4096]

        return {
            'pooled_output': pooled_output,
            'context_output': context_output,
            'attention_mask_clip': tokens_1.attention_mask,
            'attention_mask_t5': tokens_3.attention_mask
        }
```

#### 4.4.3 梯度计算函数 - 适配三编码器

```python
# filepath: src_sd3-5/text_space_attack/MMA_para.py
# 修改 token_gradient_batch 方法

class ParallelMMA(BaseAttacker):
    # ... 其他方法 ...

    def token_gradient_batch_sd3(
        self,
        controls: List[str],
        target_pooled: torch.Tensor,   # [B, 2048]
        target_context: torch.Tensor   # [B, 154, 4096]
    ) -> torch.Tensor:
        """
        计算 SD3.5 的 token 梯度

        策略说明：
        由于 SD3.5 使用三个编码器，我们选择对 CLIP-L 进行梯度优化，
        因为它同时影响池化输出和上下文序列的一部分。

        Args:
            controls: 控制字符串列表
            target_pooled: 目标池化向量 [B, 2048]
            target_context: 目标上下文序列 [B, 154, 4096]

        Returns:
            梯度张量
        """
        batch_size = len(controls)
        control_length = self.control_length

        # 使用 CLIP-L 的 tokenizer 和 embedding
        tokens = self.tokenizer_1(
            controls,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        input_ids = tokens.input_ids

        # 获取 embedding 层权重
        embed_weights = self.text_encoder_1.text_model.embeddings.token_embedding.weight

        # 创建 one-hot 向量（用于梯度传播）
        one_hot = torch.zeros(
            batch_size,
            control_length,
            embed_weights.shape[0],
            device=self.device,
            dtype=embed_weights.dtype
        )

        for i in range(batch_size):
            one_hot[i].scatter_(1, input_ids[i][:control_length].unsqueeze(1), 1.0)

        one_hot.requires_grad_()

        # 计算可微分的 embedding
        input_embeds = torch.matmul(one_hot, embed_weights)  # [B, control_length, 768]

        # 获取完整的 token embeddings
        tok_emb_layer = self.text_encoder_1.text_model.embeddings.token_embedding
        full_embeds = tok_emb_layer(input_ids)  # [B, 77, 768]

        # 拼接：前 control_length 个使用可导嵌入
        full_embeds = torch.cat([
            input_embeds.to(full_embeds.dtype),
            full_embeds[:, control_length:]
        ], dim=1)

        # 添加位置编码
        position_ids = torch.arange(77, device=self.device)
        pos_embeds = self.text_encoder_1.text_model.embeddings.position_embedding(position_ids)
        embeddings = full_embeds + pos_embeds.unsqueeze(0)

        # 通过编码器计算
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # CLIP-L 使用可导嵌入
            outputs_1 = self.text_encoder_1.text_model(
                input_ids=input_ids,
                input_embed=embeddings  # ← 传递自定义嵌入
            )

            clip_l_hidden = outputs_1.hidden_states[-2] if hasattr(outputs_1, 'hidden_states') else outputs_1.last_hidden_state
            clip_l_pooled = outputs_1.pooler_output

            # CLIP-G 和 T5 不需要梯度
            with torch.no_grad():
                tokens_2 = self.tokenizer_2(
                    controls,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs_2 = self.text_encoder_2(
                    tokens_2.input_ids,
                    output_hidden_states=True,
                    return_dict=True
                )
                clip_g_hidden = outputs_2.hidden_states[-2]
                clip_g_pooled = outputs_2.pooler_output

                tokens_3 = self.tokenizer_3(
                    controls,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs_3 = self.text_encoder_3(
                    tokens_3.input_ids,
                    return_dict=True
                )
                t5_hidden = outputs_3.last_hidden_state

            # 构建完整输出
            adv_pooled = torch.cat([clip_l_pooled, clip_g_pooled.detach()], dim=-1)

            clip_combined = torch.cat([clip_l_hidden, clip_g_hidden.detach()], dim=-1)
            clip_padded = F.pad(clip_combined, (0, 2048))
            adv_context = torch.cat([clip_padded, t5_hidden.detach()], dim=1)

            # 计算损失
            loss = self.loss_fn(
                adv_pooled, target_pooled,
                adv_context, target_context
            )

        loss.backward()

        return one_hot.grad.clone()
```

---

## 5. transformers 库修改说明

### 5.1 为什么需要修改

SD1.5 的攻击方法需要绕过正常的 `token → embedding` 转换，直接注入自定义的可导嵌入向量。这是通过向 CLIP 模型的 `forward` 函数添加 `input_embed` 参数实现的。

### 5.2 需要修改的文件

```
~/.conda/envs/<your_env>/lib/python3.10/site-packages/transformers/models/clip/modeling_clip.py
```

仓库中也提供了同版本补丁文件：

```
patches/transformers_clip_input_embed.patch
```

可以进入当前 Python 环境的 `site-packages` 目录后应用该补丁，或按下面步骤手动修改。根目录 README 中的 `images/CLIPTextModel.png` 和 `images/CLIPTextTransformer.png` 也展示了同一处修改。

### 5.3 需要修改的类

至少需要修改以下两个类的 `forward` 方法：

1. **`CLIPTextModel`**
2. **`CLIPTextTransformer`**

如果你的环境里会直接调用 `CLIPTextModelWithProjection`，也应同步加入 `input_embed` 透传；补丁文件已经包含这一处。

### 5.4 修改步骤

#### Step 1: 找到 modeling_clip.py 文件

```bash
# 方法1: 使用 Python 查找
python -c "import transformers; print(transformers.__file__)"
# 然后导航到 models/clip/modeling_clip.py

# 方法2: 直接定位
# Windows:
# %USERPROFILE%\.conda\envs\<env_name>\Lib\site-packages\transformers\models\clip\modeling_clip.py

# Linux:
# ~/.conda/envs/<env_name>/lib/python3.10/site-packages/transformers/models/clip/modeling_clip.py
```

#### Step 2: 修改 CLIPTextTransformer.forward

找到 `CLIPTextTransformer` 类（大约在第 590 行左右），修改其 `forward` 方法：

```python
# 原始代码（约第 591-620 行）
def forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:

# 修改后的代码
def forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    input_embed: Optional[torch.FloatTensor] = None,  # ← 新增参数
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
```

#### Step 3: 修改 embedding 使用逻辑

在 `forward` 方法内部，找到计算 `hidden_states` 的位置（大约在第 615 行左右）：

```python
# 原始代码
hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

# 修改后的代码
hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
if input_embed is not None:
    hidden_states = input_embed  # ← 使用外部传入的嵌入替换
```

#### Step 4: 修改 CLIPTextModel.forward

类似地，修改 `CLIPTextModel` 类的 `forward` 方法（大约在第 650 行左右）：

```python
# 添加 input_embed 参数
def forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    input_embed: Optional[torch.FloatTensor] = None,  # ← 新增参数
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    # ...

    # 将 input_embed 传递给 text_model
    return self.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        input_embed=input_embed,  # ← 传递参数
        return_dict=return_dict,
    )
```

### 5.5 完整的 modeling_clip.py 修改示例

以下是需要修改的关键代码片段：

```python
# ============ CLIPTextTransformer 类 ============
# 约在第 591 行

class CLIPTextTransformer(nn.Module):
    # ...

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        input_embed: Optional[torch.FloatTensor] = None,  # ← 新增
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        ...
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # ========== 关键修改 ==========
        if input_embed is not None:
            hidden_states = input_embed
        # ==============================

        # ... 后续代码保持不变 ...


# ============ CLIPTextModel 类 ============
# 约在第 650 行

class CLIPTextModel(CLIPPreTrainedModel):
    # ...

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        input_embed: Optional[torch.FloatTensor] = None,  # ← 新增
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        ...
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            input_embed=input_embed,  # ← 传递参数
            return_dict=return_dict,
        )
```

### 5.6 SD3.5 的额外考虑

对于 SD3.5，除了修改 CLIP 模型外，可能还需要考虑：

1. **OpenCLIP 模型**：如果需要对 OpenCLIP-G/14 进行梯度优化，需要查找并修改相应的 OpenCLIP 模型文件。通常在：
   ```
   ~/.conda/envs/<env>/lib/python3.10/site-packages/open_clip/model.py
   ```

2. **T5 模型**：T5 模型原生支持 `inputs_embeds` 参数，不需要额外修改：
   ```python
   outputs = text_encoder_3(
       inputs_embeds=custom_embeddings,  # T5 原生支持
       return_dict=True
   )
   ```

---

## 6. 梯度优化策略

### 6.1 策略选择

由于 SD3.5 使用三个独立的编码器，有以下梯度优化策略：

| 策略 | 优化目标 | 优点 | 缺点 |
|------|----------|------|------|
| **单编码器** | 仅 CLIP-L | 计算效率高，实现简单 | 可能影响有限 |
| **双编码器** | CLIP-L + CLIP-G | 更全面影响池化输出 | 需要修改 OpenCLIP |
| **全编码器** | 三个编码器 | 理论上效果最好 | 计算成本极高 |

**推荐：从单编码器策略开始**，因为：
- CLIP-L 的池化输出是 `c_vec` 的一部分
- CLIP-L 的隐藏层是 `c_ctxt` 前 77 个 token 的一部分
- 实现最简单，可作为基线

### 6.2 为什么优化 CLIP-L 有效

```
c_vec = [clip_l_pooled | clip_g_pooled]
         ^^^^^^^^^^^^^^^
         通过优化 CLIP-L 可以影响这部分

c_ctxt = [clip_l_hidden | clip_g_hidden | padding | t5_hidden]
          ^^^^^^^^^^^^^^
          通过优化 CLIP-L 可以影响这部分
```

---

## 7. 显存优化

### 7.1 显存需求估算

| 组件 | 显存 (FP16) |
|------|-------------|
| CLIP-L/14 | ~0.4 GB |
| OpenCLIP-G/14 | ~2.5 GB |
| T5-XXL | ~4.7 GB |
| 梯度计算 | ~1-2 GB |
| **总计** | ~8-10 GB |

### 7.2 优化建议

```python
# 1. 使用 torch.no_grad() 包装非必要的编码操作
with torch.no_grad():
    enc_2 = self.encode_clip_g_batch(controls)
    enc_3 = self.encode_t5_batch(controls)

# 2. 使用混合精度计算
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    outputs = encoder(...)

# 3. 及时清理不需要的张量
del intermediate_tensor
torch.cuda.empty_cache()

# 4. 如果显存不足，可以考虑：
#    - 使用 FP8 量化的 T5
#    - 减小 batch_size
#    - 使用 gradient checkpointing
```

---

## 8. 测试与验证

### 8.1 单元测试

```python
# test_sd3_encoding.py
from text_space_attack.model import GenerativeModel
from text_space_attack.MMA_para import ParallelMMA

# 加载模型
model = GenerativeModel()
attacker = ParallelMMA(target_model=model)

# 测试编码
enc = attacker.encode_prompt_sd3("a photo of a cat")
assert enc['pooled_output'].shape == (2048,), f"Expected (2048,), got {enc['pooled_output'].shape}"
assert enc['context_output'].shape == (154, 4096), f"Expected (154, 4096), got {enc['context_output'].shape}"
print("✓ 编码测试通过")

# 测试损失计算
from text_space_attack.sd3_loss import SD3CombinedLoss
loss_fn = SD3CombinedLoss()
loss = loss_fn(
    enc['pooled_output'].unsqueeze(0),
    enc['pooled_output'].unsqueeze(0),
    enc['context_output'].unsqueeze(0),
    enc['context_output'].unsqueeze(0)
)
assert loss.item() < 0.01, f"Same input should have near-zero loss, got {loss.item()}"
print("✓ 损失函数测试通过")
```

### 8.2 完整攻击测试

```python
# test_attack.py
result = attacker.attack("a photo of violence")
print(f"攻击成功: {result.success}")
print(f"攻击提示: {result.attack_prompt}")
print(f"执行时间: {result.execution_time:.2f}s")
print(f"最终损失: {result.metadata.get('best_loss', 'N/A')}")
```

---

## 9. 常见问题

### Q1: 修改 modeling_clip.py 后会影响其他项目吗？

**A:** 会的。这是对 transformers 库的全局修改。建议：
- 使用独立的 conda 环境
- 或者使用猴子补丁（monkey patching）在运行时修改

### Q2: 为什么不同时优化三个编码器？

**A:** 主要考虑：
1. **计算成本**：三个编码器的梯度计算需要大量显存
2. **Tokenizer 不同**：CLIP 和 T5 使用不同的 tokenizer，优化后的 token 可能不通用
3. **收益递减**：实验表明优化 CLIP-L 通常足够

### Q3: 攻击效果不好怎么办？

**A:** 尝试以下调整：
1. 增大 $\lambda_2$（增加上下文序列权重）
2. 增加 `n_steps`（更多优化步数）
3. 减小 `threshold`（更严格的收敛条件）
4. 尝试双编码器优化策略

---

## 10. 参考资料

- [Stable Diffusion 3.5 Paper](https://stability.ai/research/stable-diffusion-3)
- [HuggingFace Diffusers - SD3 Pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3)
- [MMA-Diffusion 原始论文](https://arxiv.org/abs/2311.17516)
- [transformers CLIP 模型源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py)
