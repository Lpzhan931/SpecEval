import time
import torch
from .base_generator import BaseGenerator

class SpSGenerator(BaseGenerator):
    def __init__(self, target_model, draft_model, tokenizer, tracker, gamma=5):
        super().__init__(tokenizer, tracker)
        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma = gamma

    def _trim_kv_cache(self, past_key_values, keep_len):
        """
        核心辅助函数：将多余的 KV Cache 截断到 keep_len 长度。
        兼容 HF 的元组格式 (Tuple) 以及新版的 DynamicCache。
        """
        if past_key_values is None:
            return None
            
        if isinstance(past_key_values, tuple):
            new_past = []
            for layer in past_key_values:
                # layer[0] 是 Key, layer[1] 是 Value
                # shape 通常为: (batch_size, num_heads, seq_len, head_dim)
                k = layer[0][:, :, :keep_len, :]
                v = layer[1][:, :, :keep_len, :]
                new_past.append((k, v))
            return tuple(new_past)
        elif hasattr(past_key_values, "key_cache"): 
            # 兼容 transformers >= 4.38 的 DynamicCache
            for i in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, :keep_len, :]
                past_key_values.value_cache[i] = past_key_values.value_cache[i][:, :, :keep_len, :]
            # 更新内置的 seen_tokens 计数器
            if hasattr(past_key_values, "_seen_tokens"):
                past_key_values._seen_tokens = keep_len
            return past_key_values
        return past_key_values

    def generate(self, input_ids, max_new_tokens, **kwargs):
        # 确保 Input 在各自模型的 Device 上
        t_device = self.target_model.device
        d_device = self.draft_model.device
        
        seq = input_ids[0].tolist()
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            # ==========================================
            # 1. Prefill
            # ==========================================
            # Target Model 预填充
            t_out = self.target_model(input_ids.to(t_device), use_cache=True)
            target_kv = t_out.past_key_values
            next_tok_t = torch.argmax(t_out.logits[0, -1, :]).item()
            
            # Draft Model 预填充
            d_out = self.draft_model(input_ids.to(d_device), use_cache=True)
            draft_kv = d_out.past_key_values
            
            # 初始的第一个预测 Token 加入序列
            seq.append(next_tok_t)
            
            generated_tokens = 1
            accepted_tokens_total = 0
            spec_steps = 0
            step_matches = []
            step_drafts = []
            
            # ==========================================
            # 2. Speculative Decoding 循环
            # ==========================================
            while generated_tokens < max_new_tokens:
                if seq[-1] == self.tokenizer.eos_token_id:
                    break
                    
                curr_token = seq[-1]
                L = len(seq) # 当前已有的总上下文长度
                
                # -------------------------
                # Step 2.1: 草稿生成 (Draft)
                # -------------------------
                draft_tokens = []
                d_in = torch.tensor([[curr_token]], device=d_device)
                
                for _ in range(self.gamma):
                    out = self.draft_model(d_in, past_key_values=draft_kv, use_cache=True)
                    draft_kv = out.past_key_values
                    d_i = torch.argmax(out.logits[0, -1, :]).item()
                    draft_tokens.append(d_i)
                    d_in = torch.tensor([[d_i]], device=d_device)
                    # 遇到 EOS 提前结束草稿
                    if d_i == self.tokenizer.eos_token_id:
                        break
                        
                actual_gamma = len(draft_tokens)
                
                # -------------------------
                # Step 2.2: 目标验证 (Verify)
                # -------------------------
                # 目标模型的输入： 当前 Token + 所有草稿 Tokens
                t_in_list = [curr_token] + draft_tokens
                t_in = torch.tensor([t_in_list], device=t_device)
                
                # 此时 target_kv 的长度是 L-1
                out = self.target_model(t_in, past_key_values=target_kv, use_cache=True)
                target_kv = out.past_key_values # 更新后长度变为 L - 1 + actual_gamma + 1
                
                # out.logits shape: (1, actual_gamma + 1, vocab_size)
                target_preds = torch.argmax(out.logits[0], dim=-1).tolist()
                
                # -------------------------
                # Step 2.3: 对齐与接受 (Match)
                # -------------------------
                k = 0
                for i in range(actual_gamma):
                    if target_preds[i] == draft_tokens[i]:
                        k += 1
                    else:
                        break
                        
                accepted_tokens_total += k
                spec_steps += 1
                step_matches.append(k)
                step_drafts.append(actual_gamma)
                
                # 本轮最终接受的 Tokens (k 个草稿 token + 1 个目标模型的纠正 token)
                tokens_to_add = draft_tokens[:k] + [target_preds[k]]
                
                # 更新序列并处理提前停止条件
                for tok in tokens_to_add:
                    seq.append(tok)
                    generated_tokens += 1
                    if tok == self.tokenizer.eos_token_id or generated_tokens >= max_new_tokens:
                        break
                
                # -------------------------
                # Step 2.4: KV Cache 截断回滚
                # -------------------------
                # 核心逻辑：保证下一次循环开始前，两个模型的 KV Cache 长度都严格等于 len(seq) - 1
                keep_len = L + k
                
                # Target 模型总是多算了，直接截断
                target_kv = self._trim_kv_cache(target_kv, keep_len)
                
                # Draft 模型的特殊情况：
                # 如果所有 draft tokens 都被接受了，且刚才并没有遇到 EOS 结束
                # 则 Draft 模型并没有计算 target_preds[k]（即 d_gamma）的 KV Cache，我们需要给它补上。
                if k == actual_gamma and seq[-1] != self.tokenizer.eos_token_id:
                    missing_d_in = torch.tensor([[draft_tokens[-1]]], device=d_device)
                    dummy_out = self.draft_model(missing_d_in, past_key_values=draft_kv, use_cache=True)
                    draft_kv = dummy_out.past_key_values
                    # 补充后，draft_kv 的长度正好为 L + gamma，等于 keep_len，无需截断。
                else:
                    draft_kv = self._trim_kv_cache(draft_kv, keep_len)

        torch.cuda.synchronize()
        end_time = time.time()
        
        # 记录指标
        self.tracker.add_spec_stats(
            time_taken=end_time - start_time,
            tokens_generated=generated_tokens,
            accepted_tokens=accepted_tokens_total,
            spec_steps=spec_steps,
            step_matches=step_matches,
            step_drafts=step_drafts
        )
        
        return torch.tensor([seq], device=input_ids.device)
