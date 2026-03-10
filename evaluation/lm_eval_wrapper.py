import torch
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

class CustomEvalWrapper(LM):
    """
    将我们的 Generator 封装进 lm-eval 的统一接口中
    """
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.tokenizer = generator.tokenizer

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """
        lm-eval 针对生成型任务 (如 GSM8K, HumanEval) 的核心入口。
        """
        res = []
        for request in requests:
            prompt = request.args[0]
            # 获取 lm-eval 传递过来的任务特定生成参数（如果没有则用默认）
            kwargs = request.args[1] if len(request.args) > 1 else {}
            max_new_tokens = kwargs.get("until", [None])[0] or 512
            if isinstance(max_new_tokens, str): # 有时 'until' 传入的是 stop words
                max_new_tokens = 512
                
            # Tokenize
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # 调用我们自己编写的 Generator
            output_ids = self.generator.generate(input_ids, max_new_tokens=max_new_tokens)
            
            # Decode (仅截取新生成的部分)
            input_len = input_ids.shape[1]
            generated_text = self.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
            
            # 处理 lm-eval 的 stop sequences (例如遇到 \n\n 停止)
            stop_sequences = kwargs.get("until", [])
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.index(stop_seq)]
                    
            res.append(generated_text)
            
        return res

    def loglikelihood(self, requests: list[Instance]):
        # 仅针对多项选择题 (如 MMLU) 时需要。
        raise NotImplementedError("This baseline currently supports generative tasks via generate_until.")

    def loglikelihood_rolling(self, requests: list[Instance]):
        raise NotImplementedError()
    