from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    def __init__(self, tokenizer, tracker):
        self.tokenizer = tokenizer
        self.tracker = tracker

    @abstractmethod
    def generate(self, input_ids, max_new_tokens, **kwargs):
        """
        接收 input_ids，返回完整的 output_ids
        必须在此函数内部调用 self.tracker 记录耗时和 Token 数
        """
        pass
    