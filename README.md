# Speculative Method Evaluation Experiments

This repository contains implementations of evaluation for several LLM decoding strategies.

## Project Structure

``` bash
.
├── config
│   └── settings.py     # configuration files
├── evaluation
│   └── lm_eval_wrapper.py
├── generation
│   ├── ar_generator.py
│   ├── base_generator.py
│   ├── medusa_generator.py
│   ├── medusa_sps_generator.py
│   └── sps_generator.py
├── main.py             # main entry point
├── models
│   ├── medusa_model.py
│   └── model_manager.py
├── README.md
└── utils
    └── metrics_tracker.py

```

## Environment


- torch
- transformers
- lm-eval


## Usage
Set the configuration in `config/settings.py` before evaluation.

``` bash
python main.py --method ar --task gsm8k --limit 10    

python main.py --method sps --task gsm8k --limit 10  

python main.py --method medusa --task gsm8k --limit 10  

python main.py --method medusa_sps --task gsm8k --limit 10  
```

