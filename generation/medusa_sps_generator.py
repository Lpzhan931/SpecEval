import time
import torch
from .base_generator import BaseGenerator

class MedusaSpSGenerator(BaseGenerator):
    def __init__(self, target_model, draft_model, medusa_sps_head, tokenizer, tracker):
        super().__init__(tokenizer, tracker)
        self.target_model = target_model
        self.draft_model = draft_model
        self.medusa_sps_head = medusa_sps_head
        self.t_device = target_model.device
        self.d_device = draft_model.device
        self.num_heads = medusa_sps_head.num_heads

    def _trim_kv_cache(self, past_key_values, keep_len):
        # 复用 sps_generator 里面的逻辑
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
            # Target Model (需要 hidden_states)
            t_out = self.target_model(input_ids.to(self.t_device), use_cache=True, output_hidden_states=True)
            target_kv = t_out.past_key_values
            last_hidden_state_t = t_out.hidden_states[-1][:, -1:, :] # [1, 1, target_dim]
            base_token = torch.argmax(t_out.logits[0, -1, :]).item()
            seq.append(base_token)
            generated_tokens += 1
            
            # Draft Model
            d_out = self.draft_model(input_ids.to(self.d_device), use_cache=True)
            draft_kv = d_out.past_key_values
            
            # ==========================================
            # 2. Speculative Loop
            # ==========================================
            while generated_tokens < max_new_tokens:
                if seq[-1] == self.tokenizer.eos_token_id:
                    break
                
                L_before = len(seq) - 1 # 当前 KV cache 应该对齐的有效长度
                
                # --------- Phase 1: Drafting ---------
                # 大模型生成多个头的隐状态
                # m_hiddens = self.medusa_sps_head.medusa_head(last_hidden_state_t) # [num_heads, 1, 1, target_dim]
                m_hiddens = [head(last_hidden_state_t) for head in self.medusa_sps_head.medusa_head]    # [num_heads, 1, 1, target_dim]
                
                draft_tokens = []
                curr_d_in = torch.tensor([[base_token]], device=self.d_device)
                
                for i in range(self.num_heads):
                    # 小模型前向，拿到 lm_head 之前的 hidden_states
                    d_out = self.draft_model(curr_d_in, past_key_values=draft_kv, use_cache=True, output_hidden_states=True)
                    draft_kv = d_out.past_key_values
                    s_hidden = d_out.hidden_states[-1] # [1, 1, draft_dim]
                    
                    # 取出对应大模型 Medusa 头特征，进行组合推理
                    m_hidden_i = m_hiddens[i]
                    sps_logits = self.medusa_sps_head.predict_token(m_hidden_i, s_hidden)
                    
                    next_tok = torch.argmax(sps_logits, dim=-1).item()
                    draft_tokens.append(next_tok)
                    
                    if next_tok == self.tokenizer.eos_token_id:
                        break
                    curr_d_in = torch.tensor([[next_tok]], device=self.d_device)
                
                actual_gamma = len(draft_tokens)
                
                # --------- Phase 2: Verification ---------
                candidates = [base_token] + draft_tokens
                t_in = torch.tensor([candidates], device=self.t_device)
                
                t_out = self.target_model(t_in, past_key_values=target_kv, use_cache=True, output_hidden_states=True)
                target_kv = t_out.past_key_values
                target_preds = torch.argmax(t_out.logits[0], dim=-1).tolist()
                
                # --------- Phase 3: Matching ---------
                match_count = 0
                for i in range(actual_gamma):
                    if target_preds[i] == draft_tokens[i]:
                        match_count += 1
                    else:
                        break
                
                accepted_tokens_total += match_count
                spec_steps += 1
                step_matches.append(match_count)
                step_drafts.append(actual_gamma)
                
                tokens_to_add = draft_tokens[:match_count] + [target_preds[match_count]]
                for tok in tokens_to_add:
                    seq.append(tok)
                    generated_tokens += 1
                    if tok == self.tokenizer.eos_token_id or generated_tokens >= max_new_tokens:
                        break
                
                if seq[-1] == self.tokenizer.eos_token_id or generated_tokens >= max_new_tokens:
                    break
                
                # --------- Phase 4: Prepare Next & Rollback KV Cache ---------
                # 拿 Target 模型最新匹配点的特征，作为下一轮大模型的起手特征
                last_accepted_idx = match_count
                last_hidden_state_t = t_out.hidden_states[-1][:, last_accepted_idx:last_accepted_idx+1, :]
                base_token = target_preds[match_count]
                
                # 裁剪对齐 KV Cache 
                keep_len = L_before + 1 + match_count
                
                target_kv = self._trim_kv_cache(target_kv, keep_len)
                
                # 兼容草稿模型末尾 KV 丢失问题（逻辑与纯 SpS 一致）
                if match_count == actual_gamma and seq[-1] != self.tokenizer.eos_token_id:
                    missing_d_in = torch.tensor([[draft_tokens[-1]]], device=self.d_device)
                    dummy_out = self.draft_model(missing_d_in, past_key_values=draft_kv, use_cache=True)
                    draft_kv = dummy_out.past_key_values
                else:
                    draft_kv = self._trim_kv_cache(draft_kv, keep_len)

        torch.cuda.synchronize()
        end_time = time.time()
        
        self.tracker.add_spec_stats(
            time_taken=end_time - start_time,
            tokens_generated=generated_tokens,
            accepted_tokens=accepted_tokens_total,
            spec_steps=spec_steps,
            step_matches=step_matches,
            step_drafts=step_drafts
        )
        return torch.tensor([seq], device=input_ids.device)

