import torch
import torch.nn as nn
from safetensors.torch import load_file
import os
import json
from pathlib import Path


class ResBlock(nn.Module):
    """Medusa 标准的残差块"""
    def __init__(self, hidden_size, hidden_size_sm=None):
        super().__init__()
        if hidden_size_sm is not None:
            self.linear = nn.Linear(hidden_size, hidden_size_sm)
        else:
            self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

        if hidden_size_sm is not None:
            self.shortcut = nn.Linear(hidden_size, hidden_size_sm, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.act(self.linear(x))

# Medusa
class MedusaHead(nn.Module):
    """一个包含多个预测头的 Medusa 模块"""
    def __init__(self, num_heads, hidden_size, num_layers=1):
        super().__init__()
        self.num_heads = num_heads
        # 创建多个预测头
        self.heads = nn.ModuleList([
            nn.Sequential(
                *[ResBlock(hidden_size) for _ in range(num_layers)]
            ) for _ in range(num_heads)
        ])

    def forward(self, hidden_states):
        """
        输入: hidden_states (batch_size, seq_len, hidden_size)
        输出: logits_list (num_heads, batch_size, seq_len, vocab_size)
        """
        medusa_logits = [head(hidden_states) for head in self.heads]
        return torch.stack(medusa_logits, dim=0)

def load_medusa_head(model_config, medusa_path, num_heads, num_layers, device, dtype):
    """加载 safetensors 权重到 MedusaHead"""
    medusa_head = MedusaHead(
        num_heads=num_heads,
        hidden_size=model_config.hidden_size,
        num_layers=num_layers
    ).to(device, dtype)
    
    state_dict = load_file(medusa_path)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # 强制修正前缀：如果 key 是以数字开头(如 '0.0.linear')，给它拼上 'heads.'
        if k[0].isdigit():
            new_key = f"heads.{k}"
        elif k.startswith("heads."):
            new_key = k
        else:
            new_key = k.replace("medusa_head.", "").replace("model.", "")
        new_state_dict[new_key] = v
        
    # 打印加载结果，确保不出现大面积 missing keys
    missing, unexpected = medusa_head.load_state_dict(new_state_dict, strict=False)
    print(f"\n[Medusa] Weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print(f"[Medusa Warning] Missing keys example: {missing[:5]}")
        
    medusa_head.eval()
    return medusa_head


# MedusaSpS
class MedusaSpSHead(nn.Module):
    """
    结合 Medusa (大模型多头推测) 与 SpS (小模型局部依赖) 的混合预测模块
    """
    def __init__(self, num_heads, num_layers, hidden_size, hidden_size_sm, vocab_size):
        super().__init__()
        self.num_heads = num_heads

        self.medusa_head = nn.ModuleList()
        for _ in range(num_heads):
            layers = []            
            layers.append(ResBlock(hidden_size, hidden_size_sm))  # 第一层降维
            for _ in range(num_layers - 1):  # 后面的层在小维度下运算
                layers.append(ResBlock(hidden_size_sm, hidden_size_sm))
            self.medusa_head.append(nn.Sequential(*layers))
        
        # 特征融合与映射层
        self.fc_layer = nn.Linear(hidden_size_sm + hidden_size_sm, hidden_size_sm)
        self.lm_head_sps = nn.Linear(hidden_size_sm, vocab_size, bias=False)
        
    def predict_token(self, m_hidden, s_hidden):
        """
        输入: 
            m_hidden: 大模型该头的输出特征 [batch, seq, hidden_size_sm]
            s_hidden: 小模型该步的隐状态 [batch, seq, hidden_size_sm]
        """
        concat_hidden = torch.cat([m_hidden, s_hidden], dim=-1)
        fc_out = m_hidden + self.fc_layer(concat_hidden)
        logits = self.lm_head_sps(fc_out)
        return logits


def load_medusa_sps_head(target_config, draft_config, path, device, dtype, fallback_heads, fallback_layers):
    """
    支持从 checkpoint/model.safetensors 或 export/medusa_sps_heads.safetensors 加载
    """
    # 判断权重文件形式
    weight_path = os.path.join(path, "model.safetensors") # 假设 path 是 '.../output_xxx/checkpoint-xxx/' 形式
    config_path = Path(path).parent / "config.json"
    if not os.path.exists(weight_path):
        weight_path = os.path.join(path, "medusa_sps_heads.safetensors") # path 是 '.../output_xxx/' 形式
        config_path = os.path.join(path, "config.json")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到权重文件！请确保 {path} 下有 model.safetensors 或 medusa_sps_heads.safetensors")
    
    # 尝试读取配置
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        num_heads = cfg.get("medusa_num_heads", fallback_heads)
        num_layers = cfg.get("medusa_num_layers", fallback_layers)
    else:
        num_heads = fallback_heads
        num_layers = fallback_layers
    
    model = MedusaSpSHead(
        num_heads=num_heads, 
        num_layers=num_layers,
        hidden_size=target_config.hidden_size,
        hidden_size_sm=draft_config.hidden_size,
        vocab_size=target_config.vocab_size
    ).to(device, dtype)
        
    state_dict = load_file(weight_path)
    
    # 清洗 keys，兼容 HF Trainer 自动包裹的 `model.` 前缀
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.medusa_head", "medusa_head") \
                   .replace("model.fc_layer", "fc_layer") \
                   .replace("model.lm_head_sps", "lm_head_sps")
        clean_state_dict[new_key] = v
        
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    
    print(f"\n[Medusa-SpS] Weights loaded from {weight_path}")
    print(f"[Medusa-SpS] num_heads: {num_heads}, num_layers: {num_layers}")
    
    # 过滤出真正缺少的组件权重（过滤掉 base_model 的权重提示）
    real_missing = [k for k in missing if k.startswith(("medusa_head", "fc_layer", "lm_head_sps"))]
    if len(real_missing) > 0:
        print(f"[Medusa-SpS WARNING] Missing essential keys: {real_missing}")
        
    model.eval()
    return model

