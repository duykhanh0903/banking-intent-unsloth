Intent Classification for Banking using Llama-3 & Unsloth

This repository contains a complete pipeline to fine-tune Llama-3 (8B) using LoRA (Low-Rank Adaptation) via the Unsloth library. The goal is to classify customer service queries into specific banking intents using the banking77 dataset.

🚀 Key Features
- **Memory Efficient**: Powered by Unsloth, enabling 2x faster training and 70% less VRAM usage.
- **Config-Driven**: Hyperparameters, LoRA ranks, and inference settings are managed via YAML files for better experiment tracking.
- **Automated Pipeline**: Includes data cleaning, label mapping, and evaluation scripts.
- **Interactive Inference**: Support for both bulk accuracy testing on CSV files and real-time interactive demo mode.

📂 Project Structure
```plaintext
├── configs/
│   ├── train.yaml          # Training & LoRA hyperparameters
│   └── inference.yaml      # Inference settings & prompt templates
├── model/
│   └── intent-banking-model/  # Default output directory for fine-tuned adapter
├── sample_data/
│   ├── label.csv           # Label name mapping for reference
│   ├── train.csv           # Training data
│   ├── test.csv            # Test data
│   └── val.csv             # Validation data
├── scripts/
│   ├── preprocess_data.py  # Script for data cleaning and splitting
│   ├── train.py            # Main training script (Fine-tuning)
│   └── inference.py        # Accuracy testing & interactive prediction
├── train.sh                # Shell script wrapper for training (optional)
├── inference.sh            # Shell script wrapper for inference (optional)
└── requirements.txt        # Python dependencies
```

🛠 Setup & Installation

Local Environment

It is recommended to use Python 3.10+ and a CUDA-enabled GPU.

```bash
pip install -r requirements.txt
```

**Note**: For the best performance, follow the Unsloth installation guide to match your specific CUDA version.

☁️ Running on Google Colab

You can also run this project entirely on Google Colab. Follow the steps below:

1. **Upload the repository** to your Google Drive or directly upload the folder to the Colab session.
2. **Mount Google Drive** (if applicable):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Navigate to the project directory**:
   ```bash
   %cd /content/drive/MyDrive/banking-intent-unsloth
   # Or if uploaded directly to the session:
   # %cd /content/banking-intent-unsloth
   ```
4. **Install dependencies**:
   ```bash
   !pip install -r requirements.txt
   # Unsloth often requires a specific PyTorch/CUDA version on Colab.
   # If you encounter errors, run:
   # !pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
   # !pip install --no-deps cut_cross_entropy unsloth_zoo
   # !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
   # !pip install --no-deps unsloth
   ```
5. **Run preprocessing**:
   ```bash
   !python scripts/preprocess_data.py
   ```
6. **Run training**:
   ```bash
   !python scripts/train.py
   ```
7. **Run inference**:
   ```bash
   !python scripts/inference.py
   ```

**Colab Tip**: Make sure your runtime is set to **T4 GPU** (or higher) via `Runtime > Change runtime type > Hardware accelerator > GPU`.

🗃 Data Preparation

Run the preprocessing script to download, clean, and split the banking77 dataset:

```bash
python scripts/preprocess_data.py
```

**What this script does**:
- Loads the `banking77` dataset from Hugging Face.
- Performs stratified sampling: **40%** of train set, **15%** of test set, then splits **10%** from train for validation.
- Cleans text: lowercasing, removing special characters, and normalizing whitespace.
- Maps numeric labels to human-readable intent names (e.g., `0` → `activate_my_card`).
- Saves the processed files to `sample_data/`.

🚄 Training

Adjust the parameters in `configs/train.yaml` (e.g., `learning_rate`, `num_train_epochs`, `r`) then run:

```bash
python scripts/train.py
```

The fine-tuned adapter will be saved to `model/intent-banking-model` by default.

**Training highlights** (see `configs/train.yaml`):
- **Model**: `unsloth/llama-3-8b-bnb-4bit` with 4-bit quantization.
- **RS-LoRA**: `use_rslora: true` for better scaling with large ranks.
- **Packing**: `packing: true` for efficient sequence utilization.
- **Gradient Checkpointing**: `use_gradient_checkpointing: "unsloth"` to save VRAM.
- **Optimizer**: `adamw_8bit` for memory-efficient training.
- **Evaluation**: Performed every `300` steps, saving the best model based on `eval_loss`.

🧪 Inference & Evaluation

1. Accuracy Test

The script will evaluate the model against `sample_data/test.csv` and generate an accuracy score:

```bash
python scripts/inference.py
```

If there are mismatches, the script automatically exports them to `test_errors.csv` for further analysis.

2. Interactive Demo (User Input)

You can test the model with custom queries by following the on-screen prompt when running the inference script in interactive mode, or by modifying the `if __name__ == "__main__":` block in `scripts/inference.py` to use `input()`:

```python
# Example Usage:
# User: "I am still waiting for my new card to arrive."
# Output: Predicted Label: card_arrival
```

📝 Configuration

train.yaml

Used to define the model, LoRA architecture, and training hyperparameters:

| Section | Key | Description |
|---------|-----|-------------|
| `model` | `model_name` | Base model ID from Hugging Face (`unsloth/llama-3-8b-bnb-4bit`). |
| `model` | `max_seq_length` | Maximum sequence length (`256`). |
| `model` | `load_in_4bit` | Enable 4-bit quantization (`true`). |
| `lora` | `r` | Rank of the update matrices (`16`). |
| `lora` | `target_modules` | Specific layers to apply LoRA (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). |
| `lora` | `lora_alpha` | Scaling factor for LoRA (`32`). |
| `lora` | `lora_dropout` | Dropout rate for LoRA layers (`0`). |
| `lora` | `bias` | Bias training mode (`none`). |
| `lora` | `use_rslora` | Use RS-LoRA for better convergence (`true`). |
| `training` | `per_device_train_batch_size` | Batch size per GPU (`2`). |
| `training` | `gradient_accumulation_steps` | Gradient accumulation steps (`16`). |
| `training` | `warmup_steps` | Number of warmup steps (`10`). |
| `training` | `learning_rate` | Initial learning rate (`0.0001`). |
| `training` | `num_train_epochs` | Total training epochs (`3`). |
| `training` | `optim` | Optimizer (`adamw_8bit`). |
| `training` | `lr_scheduler_type` | Learning rate scheduler (`cosine`). |
| `training` | `evaluation_strategy` | When to evaluate (`steps`). |
| `training` | `eval_steps` | Evaluate every N steps (`300`). |
| `training` | `save_strategy` | When to save (`steps`). |
| `training` | `save_steps` | Save every N steps (`300`). |
| `training` | `load_best_model_at_end` | Load best checkpoint at end (`true`). |
| `training` | `metric_for_best_model` | Metric to select best model (`eval_loss`). |
| `training` | `packing` | Enable sequence packing (`true`). |
| `training` | `use_gradient_checkpointing` | Gradient checkpointing strategy (`unsloth`). |
| `training` | `output_dir` | Directory to save adapter (`model/intent-banking-model`). |

inference.yaml

Used to control how the model generates and parses answers:

| Section | Key | Description |
|---------|-----|-------------|
| - | `model_base` | Base model ID (`unsloth/llama-3-8b-bnb-4bit`). |
| - | `model_path` | Path to the fine-tuned adapter (`model/intent-banking-model`). |
| - | `max_seq_length` | Maximum sequence length (`256`). |
| - | `load_in_4bit` | Use 4-bit quantization (`true`). |
| - | `device` | Device to run inference on (`cuda`). |
| `generation` | `max_new_tokens` | Maximum tokens to generate (`20`). |
| `generation` | `do_sample` | Whether to use sampling (`false`). |
| `generation` | `temperature` | Sampling temperature (`0.1`). |
| `postprocess` | `keyword` | Anchor string to extract the predicted label (`The correct answer is: class`). |
| - | `prompt_template` | Prompt template used for inference. Must match the training format. |

⚠️ Prompt Alignment

It is critical that the `prompt_template` in `configs/inference.yaml` matches the prompt used during training in `scripts/train.py`. Any mismatch will lead to incorrect or unpredictable outputs.

**Training prompt format** (in `scripts/train.py`):
```plaintext
Here is a customer query:
{input}

Classify this into one of the following intent:

The correct answer is: class {label}
```

**Inference prompt format** (in `configs/inference.yaml`):
```yaml
prompt_template: |
  Here is a customer query:
  {input}

  Classify this into one of the following intent:

  The correct answer is: class
```

The only difference should be the absence of the `{label}` placeholder during inference, as the model is expected to generate it.

Link demo and model: https://drive.google.com/drive/folders/1ct5NBemHwZEMPCUdkbcCE2-oBtRO0Frt?usp=sharing