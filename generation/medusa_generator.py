import time
import torch
from .base_generator import BaseGenerator

class MedusaGenerator(BaseGenerator):
    def __init__(self, target_model, medusa_head, tokenizer, tracker):
        super().__init__(tokenizer, tracker)
        self.model = target_model
        self.medusa_head = medusa_head
        self.device = target_model.device
        self.num_heads = medusa_head.num_heads

    def _trim_kv_cache(self, past_key_values, keep_len):
        """
        同 SpS, 辅助函数：将多余的 KV Cache 截断到 keep_len 长度。
        兼容 HF 的元组格式 (Tuple) 以及新版的 DynamicCache。
        """
        if past_key_values is None: return None
        if isinstance(past_key_values, tuple):
            new_past = []
            for layer in past_key_values:
                k = layer[0][:, :, :keep_len, :]
                v = layer[1][:, :, :keep_len, :]
                new_past.append((k, v))
            return tuple(new_past)
        elif hasattr(past_key_values, "key_cache"): 
            for i in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, :keep_len, :]
                past_key_values.value_cache[i] = past_key_values.value_cache[i][:, :, :keep_len, :]
            if hasattr(past_key_values, "_seen_tokens"):
                past_key_values._seen_tokens = keep_len
            return past_key_values
        return past_key_values

    def generate(self, input_ids, max_new_tokens, **kwargs):
        input_ids = input_ids.to(self.device)
        seq = input_ids[0].tolist()
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        generated_tokens = 0
        accepted_tokens_total = 0
        spec_steps = 0
        step_matches = []
        step_drafts = []
        
        with torch.no_grad():
            # ==========================================
            # 1. Prefill 阶段
            # ==========================================
            # 注意：必须传入 output_hidden_states=True
            outputs = self.model(
                input_ids, 
                use_cache=True, 
                output_hidden_states=True
            )
            past_key_values = outputs.past_key_values
            
            # 拿到 Prompt 最后一个 Token 的 Hidden State
            last_hidden_state = outputs.hidden_states[-1][:, -1:, :] # shape: [1, 1, hidden_size]
            
            # Base Model 预测出第一个 Next Token (c_0)
            base_token = torch.argmax(outputs.logits[0, -1, :]).item()
            seq.append(base_token)
            generated_tokens += 1
            
            # Medusa Heads 预测后续的 N 个 Tokens (c_1, c_2, ..., c_N)
            medusa_states = self.medusa_head(last_hidden_state) # shape: [num_heads, 1, 1, hidden_size]
            medusa_logits = self.model.lm_head(medusa_states)   # shape: [num_heads, 1, 1, vocab_size]
            medusa_tokens = torch.argmax(medusa_logits[:, 0, 0, :], dim=-1).tolist()
            
            # ==========================================
            # 2. Medusa 解码循环
            # ==========================================
            while generated_tokens < max_new_tokens:
                if seq[-1] == self.tokenizer.eos_token_id:
                    break

                # 注意此时 seq 包含 base_token
                L = len(seq)
                
                # 当前要送入大模型验证的序列链：[base_token, medusa_1, medusa_2, ...]
                candidates = [base_token] + medusa_tokens   # [1 + N, ]
                candidate_tensor = torch.tensor([candidates], device=self.device)   # [1, 1 + N]
                
                # Base Model 前向传播进行验证
                outputs = self.model(
                    candidate_tensor,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                
                past_key_values = outputs.past_key_values
                # outputs.logits shape: [1, num_candidates, vocab_size]
                target_preds = torch.argmax(outputs.logits[0], dim=-1).tolist()
                
                # -------------------------
                # 验证匹配逻辑
                # -------------------------
                # target_preds[i] 是模型基于 candidates[0..i] 预测的下一个 token。
                # 所以我们期望 target_preds[i] == candidates[i+1] (即 Medusa 的预测)
                match_count = 0
                for i in range(len(medusa_tokens)):
                    if target_preds[i] == candidates[i+1]:
                        match_count += 1
                    else:
                        break
                        
                accepted_tokens_total += match_count
                spec_steps += 1
                step_matches.append(match_count)
                step_drafts.append(len(medusa_tokens))

                # # TODO 作弊测试
                # cheat_match_count = len(medusa_tokens)
                # match_count = cheat_match_count
                # accepted_tokens_total += match_count
                # spec_steps += 1

                
                # 本轮最终接受的 Tokens：成功匹配的 Medusa Tokens + 1个纠正/新预测的 Token
                # tokens_to_add = [candidates[0]] + medusa_tokens[:match_count] + [target_preds[match_count]]
                tokens_to_add = medusa_tokens[:match_count] + [target_preds[match_count]]
                
                # 更新序列
                for tok in tokens_to_add:
                    seq.append(tok)
                    generated_tokens += 1
                    if tok == self.tokenizer.eos_token_id or generated_tokens >= max_new_tokens:
                        break
                
                if seq[-1] == self.tokenizer.eos_token_id or generated_tokens >= max_new_tokens:
                    break
                    
                # -------------------------
                # 准备下一轮的 Draft
                # -------------------------
                # 刚才验证通过的最后那个 Token 是 target_preds[match_count]。
                # 它的前驱特征（生成它的特征）在 outputs.hidden_states[-1][0, match_count, :] 中。
                # 我们利用这个特征再次触发 Medusa 和 Base 预测下一轮的 candidates。
                
                last_accepted_idx = match_count
                last_hidden_state = outputs.hidden_states[-1][:, last_accepted_idx:last_accepted_idx+1, :]
                
                # 下一轮的起始 token 就是刚刚纠正/新预测出来的 token
                base_token = target_preds[match_count]
                
                # 再次触发 Medusa
                medusa_states = self.medusa_head(last_hidden_state)
                medusa_logits = self.model.lm_head(medusa_states)
                medusa_tokens = torch.argmax(medusa_logits[:, 0, 0, :], dim=-1).tolist()
                
                # -------------------------
                # KV Cache 截断
                # -------------------------
                # 我们刚刚喂入了 len(candidates) 个 token，但只保留了 match_count + 1 个状态
                # (候选的 c_0 到 c_{match_count}, 注意 L 这个长度包括 base_token)
                # 需要丢弃多余的 KV Cache
                keep_len = L + match_count
                past_key_values = self._trim_kv_cache(past_key_values, keep_len)

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
        
        return torch.tensor([seq], device=self.device)
