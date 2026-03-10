import time
from collections import defaultdict


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_generated_tokens = 0
        self.total_wall_time = 0.0
        
        # Spec 整体指标
        self.num_speculative_steps = 0
        self.total_accepted_tokens = 0

        # Positional Acceptance Rate
        self.position_attempts = defaultdict(int)
        self.position_successes = defaultdict(int)

    def add_ar_stats(self, time_taken, tokens_generated):
        """记录标准自回归(ar)的统计信息"""
        self.total_wall_time += time_taken
        self.total_generated_tokens += tokens_generated

    def add_spec_stats(self, time_taken, tokens_generated, accepted_tokens, spec_steps, step_matches=None, step_drafts=None):
        """记录 spec 方法的统计信息"""
        self.total_wall_time += time_taken
        self.total_generated_tokens += tokens_generated
        self.total_accepted_tokens += accepted_tokens
        self.num_speculative_steps += spec_steps

        # positional acceptance
        if step_matches is not None and step_drafts is not None:
            for m, n in zip(step_matches, step_drafts):
                # m = match_count (匹配的长度), n = actual_gamma (推测的长度)
                # 位置从 1 开始
                for i in range(1, m + 1):
                    self.position_attempts[i] += 1
                    self.position_successes[i] += 1
                
                # 第 m+1 个 token 尝试了，但是失败了
                if m < n:
                    self.position_attempts[m + 1] += 1

    # def print_summary(self, method="ar"):
    #     print("\n" + "="*40)
    #     print(f"[{method}] Generation Metrics Summary:")
    #     if self.total_wall_time > 0:
    #         speed = self.total_generated_tokens / self.total_wall_time
    #         print(f"Total Tokens Generated : {self.total_generated_tokens}")
    #         print(f"Total Wall Time        : {self.total_wall_time:.2f} s")
    #         print(f"Generation Speed       : {speed:.2f} tokens/sec")
        
    #     if method != "ar" and self.num_speculative_steps > 0:
    #         mean_acceptance = self.total_accepted_tokens / self.num_speculative_steps
    #         print(f"Speculative Steps      : {self.num_speculative_steps}")
    #         print(f"Total Accepted Tokens  : {self.total_accepted_tokens}")
    #         print(f"Mean Acceptance Length : {mean_acceptance:.2f} tokens/step")
    #     print("="*40 + "\n")
    
    def print_summary(self, method="ar"):
        print("\n" + "="*50)
        print(f"[{method.upper()}] Generation Metrics Summary:")
        if self.total_wall_time > 0:
            speed = self.total_generated_tokens / self.total_wall_time
            print(f"Total Tokens Generated : {self.total_generated_tokens}")
            print(f"Total Wall Time        : {self.total_wall_time:.2f} s")
            print(f"Generation Speed       : {speed:.2f} tokens/sec")
        
        if method in ["sps", "medusa", "medusa_sps"] and self.num_speculative_steps > 0:
            mean_acceptance = self.total_accepted_tokens / self.num_speculative_steps
            print(f"Speculative Steps      : {self.num_speculative_steps}")
            print(f"Total Accepted Tokens  : {self.total_accepted_tokens}")
            print(f"Mean Acceptance Length : {mean_acceptance:.2f} tokens/step")
            
            # positional acceptance rate
            print("-" * 50)
            print("Acceptance Rate Per Position:")
            positions = sorted(self.position_attempts.keys())
            for pos in positions:
                attempts = self.position_attempts[pos]
                successes = self.position_successes[pos]
                rate = successes / attempts if attempts > 0 else 0.0
                print(f"  Token {pos}: {rate:6.2%} ({successes}/{attempts})")
                
        print("="*50 + "\n")
        
        