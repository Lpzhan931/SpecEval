import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import Config
from .medusa_model import load_medusa_head, load_medusa_sps_head


class ModelManager:
    def __init__(self, load_target=True, load_draft=False, load_medusa=False, load_medusa_sps=False):
        self.tokenizer = None
        self.target_model = None
        self.draft_model = None
        self.medusa_head = None
        self.medusa_sps_head = None
        
        print("Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TARGET_MODEL_PATH, trust_remote_code=True)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if load_target:
            print(f"Loading Target Model from {Config.TARGET_MODEL_PATH} ...")
            self.target_model = AutoModelForCausalLM.from_pretrained(
                Config.TARGET_MODEL_PATH,
                torch_dtype=Config.DTYPE,
                device_map="auto", 
                trust_remote_code=True
            ).eval()

        if load_draft or load_medusa_sps:
            print(f"Loading Draft Model from {Config.DRAFT_MODEL_PATH} ...")
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                Config.DRAFT_MODEL_PATH,
                torch_dtype=Config.DTYPE,
                device_map="auto",
                trust_remote_code=True
            ).eval()

        if load_medusa and self.target_model is not None:
            print(f"Loading Medusa Head from {Config.MEDUSA_HEAD_PATH} ...")
            self.medusa_head = load_medusa_head(
                self.target_model.config,
                Config.MEDUSA_HEAD_PATH,
                Config.MEDUSA_NUM_HEADS,
                Config.MEDUSA_NUM_LAYERS,
                self.target_model.device,
                Config.DTYPE
            )
        
        if load_medusa_sps and self.target_model is not None and self.draft_model is not None:
            print(f"Loading Medusa-SpS Head from {Config.MEDUSA_SPS_PATH} ...")
            self.medusa_sps_head = load_medusa_sps_head(
                target_config=self.target_model.config,
                draft_config=self.draft_model.config,
                path=Config.MEDUSA_SPS_PATH,
                device=self.target_model.device,
                dtype=Config.DTYPE,
                fallback_heads=Config.MEDUSA_SPS_FALLBACK_HEADS,
                fallback_layers=Config.MEDUSA_SPS_FALLBACK_LAYERS
            )

    def cleanup(self):
        """释放显存"""
        del self.target_model
        del self.draft_model
        del self.medusa_head
        del self.medusa_sps_head
        torch.cuda.empty_cache()
        