import yaml
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

with open("configs/train.yaml", "r") as f:
    config = yaml.safe_load(f)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = config['model']['model_name'],
    max_seq_length = config['model']['max_seq_length'],
    load_in_4bit = config['model']['load_in_4bit'],
)

model = FastLanguageModel.get_peft_model(
    model,
    r = config['lora']['r'],
    target_modules = config['lora']['target_modules'],
    lora_alpha = config['lora']['lora_alpha'],
    lora_dropout = config['lora']['lora_dropout'],
    bias = config['lora']['bias'],
    use_gradient_checkpointing = config['training']['use_gradient_checkpointing'], 
    random_state = config['training']['seed'],
    use_rslora = config['lora']['use_rslora'],
)


prompt = """Here is a customer query:
{}

Classify this into one of the following intent:

The correct answer is: class {}"""

EOS_TOKEN = tokenizer.eos_token 

def formatting_prompts_func(examples):
    inputs  = examples["text"]  
    outputs = examples["label"] 
    texts = []
    for input_text, output_label in zip(inputs, outputs):
        text = prompt.format(input_text, output_label) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("csv", data_files={
    "train": "sample_data/train.csv", 
    "val": "sample_data/val.csv"
})
dataset = dataset.map(formatting_prompts_func, batched = True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["val"], 
    dataset_text_field = "text",
    max_seq_length = config['model']['max_seq_length'],
    args = TrainingArguments(
        per_device_train_batch_size = config['training']['per_device_train_batch_size'], 
        gradient_accumulation_steps = config['training']['gradient_accumulation_steps'],
        warmup_steps = config['training']['warmup_steps'],
        learning_rate = config['training']['learning_rate'],
        num_train_epochs = config['training']['num_train_epochs'],
        optim = config['training']['optim'],
        weight_decay = config['training']['weight_decay'],
        lr_scheduler_type = config['training']['lr_scheduler_type'],
        seed = config['training']['seed'],
        output_dir = config['training']['output_dir'],
        
        eval_strategy = config['training']['evaluation_strategy'],
        eval_steps = config['training']['eval_steps'],
        save_strategy = config['training']['save_strategy'],
        save_steps = config['training']['save_steps'], 
        load_best_model_at_end = config['training']['load_best_model_at_end'],
        metric_for_best_model = config['training']['metric_for_best_model'],
        
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = config['training']['logging_steps'],
        report_to = config['training']['report_to'],
    ),
)


trainer.train()

model.save_pretrained(config['training']['output_dir'])
tokenizer.save_pretrained(config['training']['output_dir'])