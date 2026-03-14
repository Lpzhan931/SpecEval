import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import Config
from .medusa_model import load_medusa_head, load_medusa_sps_head
import copy


class ModelManager:
    def __init__(
        self, 
        load_target=True, 
        load_draft=False, 
        load_medusa=False, 
        load_medusa_sps=False,
        draft_from_trainable_param=False
    ):
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
            print(f"Loading Draft Model Base Architecture from {Config.DRAFT_MODEL_PATH} ...")
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                Config.DRAFT_MODEL_PATH,
                torch_dtype=Config.DTYPE,
                device_map="auto",
                trust_remote_code=True
            ).eval()

        # TODO DEBUG
        original_model = copy.deepcopy(self.draft_model)

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
            print(f"Loading MAR Params from {Config.MEDUSA_SPS_PATH} ...")
            self.medusa_sps_head = load_medusa_sps_head(
                target_config=self.target_model.config,
                draft_config=self.draft_model.config,
                path=Config.MEDUSA_SPS_PATH,
                device=self.target_model.device,
                dtype=Config.DTYPE,
                fallback_heads=Config.MEDUSA_SPS_FALLBACK_HEADS,
                fallback_layers=Config.MEDUSA_SPS_FALLBACK_LAYERS,
                draft_model=self.draft_model,
                draft_from_trainable_param=draft_from_trainable_param
            )

        # TODO DEBUG
        total_diff = sum(
            (p1 - p2).abs().sum().item()
            for p1, p2 in zip(original_model.parameters(), self.draft_model.parameters())
        )
        print(f"总绝对差异: {total_diff:.6f}")
        del original_model


    def cleanup(self):
        """释放显存"""
        del self.target_model
        del self.draft_model
        del self.medusa_head
        del self.medusa_sps_head
        torch.cuda.empty_cache()
        