import time
import torch
from .base_generator import BaseGenerator

class ARGeneratorV0(BaseGenerator):
    def __init__(self, model, tokenizer, tracker):
        super().__init__(tokenizer, tracker)
        self.model = model
        self.device = model.device

    def generate(self, input_ids, max_new_tokens, **kwargs):
        input_ids = input_ids.to(self.device)
        input_len = input_ids.shape[1]

        # 确保是 Greedy Decoding
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # 仅测量真正的生成过程耗时
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gen_kwargs)
            
        torch.cuda.synchronize()
        end_time = time.time()

        # 统计生成的 Token 数 (剔除 prompt 部分)
        generated_len = output_ids.shape[1] - input_len
        
        # 记录到 Tracker
        self.tracker.add_ar_stats(
            time_taken=(end_time - start_time),
            tokens_generated=generated_len
        )

        return output_ids
    
class ARGenerator(BaseGenerator):
    def __init__(self, model, tokenizer, tracker):
        super().__init__(tokenizer, tracker)
        self.model = model
        self.device = model.device

    def generate(self, input_ids, max_new_tokens, **kwargs):
        """
        手工实现 Greedy 自回归生成确保与 SpS 的底层逻辑开销对齐
        """
        input_ids = input_ids.to(self.device)
        seq = input_ids[0].tolist() 
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            # ==========================================
            # 1. Prefill
            # ==========================================
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # 获取第一个预测的 Token
            next_token = torch.argmax(outputs.logits[0, -1, :]).item()
            seq.append(next_token)
            
            generated_tokens = 1
            
            # ==========================================
            # 2. Decode
            # ==========================================
            while generated_tokens < max_new_tokens:
                # 遇到 EOS token，提前终止
                if next_token == self.tokenizer.eos_token_id:
                    break
                
                # 构建当前步的输入，形状为 [1, 1]
                current_input = torch.tensor([[next_token]], device=self.device)
                
                # 前向传播，传入 past_key_values，模型内部会自动处理 Attention 偏移
                outputs = self.model(current_input, past_key_values=past_key_values, use_cache=True)
                
                # 更新 KV Cache
                past_key_values = outputs.past_key_values
                
                # 获取下一个 Token
                next_token = torch.argmax(outputs.logits[0, -1, :]).item()
                seq.append(next_token)
                
                generated_tokens += 1
                
        torch.cuda.synchronize()
        end_time = time.time()

        # 记录到 Tracker
        self.tracker.add_ar_stats(
            time_taken=(end_time - start_time),
            tokens_generated=generated_tokens
        )

        # 返回符合 lm-eval wrapper 期望形状的 tensor: [batch_size=1, seq_len]
        return torch.tensor([seq], device=input_ids.device)
