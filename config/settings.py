import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16

    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.0      # Greedy Decoding (T=0)

    # Base Model
    TARGET_MODEL_PATH = "/home/share/models/Qwen3-8B"

    # 1. SpS Draft Model
    # DRAFT_MODEL_PATH = "/home/share/models/Llama-3.2-1B"
    # DRAFT_MODEL_PATH = "/home/share/models/Qwen2.5-0.5B"
    DRAFT_MODEL_PATH = "/home/share/models/Qwen3-0.6B"

    GAMMA = 4              # 每次 Draft 生成的 token 数量

    # 2. Medusa
    MEDUSA_NUM_LAYERS = 1 

    # trained with 40K samples
    # MEDUSA_HEAD_PATH = "/home/pzli/Project/Spec/medusa/Medusa/output_qwen3_20260305_211249_medusa_mlp__medusa_4_lr_0.001_layers_1/medusa_lm_head.safetensors"
    # MEDUSA_NUM_HEADS = 4 

    # trained with 40K samples
    # MEDUSA_HEAD_PATH = "/home/pzli/Project/Spec/medusa/Medusa/output_qwen3_test_2131_medusa_mlp__medusa_3_lr_0.001_layers_1/medusa_lm_head.safetensors"
    # MEDUSA_NUM_HEADS = 3 

    # trained with 128 samples
    # MEDUSA_HEAD_PATH = "/home/pzli/Project/Spec/medusa/Medusa/output_qwen3_2026_0306_2118_medusa_mlp__medusa_4_lr_0.001_layers_1/medusa_lm_head.safetensors"
    # MEDUSA_NUM_HEADS = 4
    
    # trained with 80K samples
    # MEDUSA_HEAD_PATH = "/home/pzli/Project/Spec/medusa/Medusa/output_qwen3_20260306_224558_medusa_mlp__medusa_4_lr_0.001_layers_1/medusa_lm_head.safetensors"
    # MEDUSA_NUM_HEADS = 4 

    # trained with 40K samples 4 epochs
    MEDUSA_HEAD_PATH = "/home/pzli/Project/Spec/medusa/Medusa/output_qwen3_20260307_211803_medusa_mlp__medusa_1_lr_0.001_layers_1/medusa_lm_head.safetensors"
    MEDUSA_NUM_HEADS = 1



    # 2. Medusa_SpS
    # MEDUSA_SPS_PATH = "/home/pzli/Project/Spec/SpS/260309/output/output_qwen3_20260309_1606_medusa_mlp__medusa_4_lr_0.001_layers_1/checkpoint-50/"
    MEDUSA_SPS_PATH = "/home/pzli/Project/Spec/SpS/260309/output/output_qwen3_20260309_184343_medusa_mlp__medusa_4_lr_0.001_layers_1"
    # 如果加载的是单独的 safetensors 且没有 config.json，则使用以下备用参数
    MEDUSA_SPS_FALLBACK_HEADS = 4
    MEDUSA_SPS_FALLBACK_LAYERS = 1
