import argparse
import sys
from lm_eval import evaluator, tasks
from models.model_manager import ModelManager
from utils.metrics_tracker import MetricsTracker
from generation.ar_generator import ARGenerator
from generation.sps_generator import SpSGenerator
from generation.medusa_generator import MedusaGenerator
from generation.medusa_sps_generator import MedusaSpSGenerator
from evaluation.lm_eval_wrapper import CustomEvalWrapper
from config.settings import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Speculative Decoding Baseline")
    parser.add_argument("--method", type=str, choices=["ar", "sps", "medusa", "medusa_sps"], default="ar",
                        help="Decoding method: 'ar' (Autoregressive) or 'sps' (Speculative Decoding) or 'medusa' (Medusa-1 chain-based) or 'medusa_sps'")
    parser.add_argument("--task", type=str, default="gsm8k",
                        help="lm-eval task name (e.g., gsm8k, humaneval)")
    parser.add_argument("--num_fewshot", type=int, default=0,
                        help="Number of few-shot examples for lm-eval")
    parser.add_argument("--limit", type=int, default=5,
                        help="Limit the number of evaluation samples (for fast testing)")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"--- Starting Task: {args.task} | Method: {args.method} ---")

    # 1. 初始化 Tracker
    tracker = MetricsTracker()

    # 2. 根据方法加载模型 (AR 只需 target，SpS 需要 target + draft, Medusa 需要 target + medusa_head)
    load_draft = (args.method in ["sps", "medusa_sps"])
    load_medusa = (args.method == "medusa")
    load_medusa_sps = (args.method == "medusa_sps")
    model_manager = ModelManager(
        load_target=True, 
        load_draft=load_draft, 
        load_medusa=load_medusa,
        load_medusa_sps=load_medusa_sps
    )

    # 3. 初始化 Generator
    if args.method == "ar":
        generator = ARGenerator(
            model=model_manager.target_model,
            tokenizer=model_manager.tokenizer,
            tracker=tracker
        )
    elif args.method == "sps":
        generator = SpSGenerator(
            target_model=model_manager.target_model,
            draft_model=model_manager.draft_model,
            tokenizer=model_manager.tokenizer,
            tracker=tracker,
            gamma=Config.GAMMA
        )
    elif args.method == "medusa":
        generator = MedusaGenerator(
            target_model=model_manager.target_model,
            medusa_head=model_manager.medusa_head,
            tokenizer=model_manager.tokenizer,
            tracker=tracker
        )
    elif args.method == "medusa_sps":
        generator = MedusaSpSGenerator(
            target_model=model_manager.target_model,
            draft_model=model_manager.draft_model,
            medusa_sps_head=model_manager.medusa_sps_head,
            tokenizer=model_manager.tokenizer,
            tracker=tracker
        )

    # 4. 封装进 lm-eval 的 Wrapper
    eval_lm = CustomEvalWrapper(generator)

    # 5. 启动 lm-eval 评测
    print(f"\nRunning evaluation on task: {args.task} ...")
    
    # 获取 lm-eval 内部的任务字典
    task_manager = tasks.TaskManager()
    
    results = evaluator.simple_evaluate(
        model=eval_lm,
        tasks=[args.task],
        num_fewshot=args.num_fewshot,
        limit=args.limit, # 限制样本数以便快速测试，跑全量设为 None
        task_manager=task_manager
    )

    # 6. 打印结果
    print("\n" + "-"*30 + " Accuracy/Task Metrics " + "-"*30)
    if results is not None:
        # 格式化输出 lm-eval 字典中的结果
        for task_name, task_res in results['results'].items():
            print(f"Task: {task_name}")
            for metric, value in task_res.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
    
    # 7. 打印速度与 SpS/Medusa 指标
    tracker.print_summary(method=args.method)

if __name__ == "__main__":
    main()
    