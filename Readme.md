# Sakhi - Telugu language model

A transformer-based language model pretrained from scratch on a cleaned and deduplicated Telugu corpus. It is trained on high-quality, natural Telugu text collected from diverse sources.

---

- **Model Availability:** [Hugging Face – abhi11nav/sakhi-telugu-681M-pretrained-0625](https://huggingface.co/abhi11nav/sakhi-telugu-681M-pretrained-0625)

## Inference

To run inference, use the following code:

> **Note:** This is a **pretrained** model only. It is not instruction-tuned or fine-tuned for specific downstream tasks.

```python
import torch

from sakhi import load_model, load_tokenizer
from sakhi.pipelines.inference.inference import generate_text

if __name__ == "__main__":
    model_path = "abhi11nav/sakhi-telugu-681M-pretrained-0625"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(path=model_path)
    model = load_model(path=model_path, device=device)

    input_text = "హలో"
    text = generate_text(
        prompt=input_text,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        device=device,
    )

    print(text)
```

## Dataset Preparation

### Datasets Used

- [`ai4bharat/sangraha`](https://huggingface.co/datasets/ai4bharat/sangraha)
- [`allenai/c4`](https://huggingface.co/datasets/allenai/c4)
- [`oscar-corpus/oscar`](https://huggingface.co/datasets/oscar-corpus/oscar)

The training corpus was carefully prepared using the following steps to ensure data quality, linguistic relevance, and uniqueness:

### 1. Data Filtering

- From **AI4Bharat/Sangraha**, only Telugu-native content was selected. Synthetic dataset was **excluded**.
- From **allenai/c4** and **oscar**, only documents identified as Telugu language were retained.

### 2. Cleaning & Deduplication Pipeline

A custom deduplication and cleaning pipeline was developed using `MinHash` and `Locality Sensitive Hashing (LSH)` to eliminate near-duplicate documents and maintain a diverse dataset.

**Steps included:**

- **Text Normalization**:

  - Stripping extra whitespaces.
  - Replacing multiple newlines and tabs with a single space.

- **MinHash-based Deduplication**:
  - A `MinHashLSH` index was used with:
    - `num_perm = 128`
    - `similarity_threshold = 0.95`
  - Each document was tokenized at the word level and hashed.
  - Duplicates were detected and removed without adding them to the final corpus.

## Model Parameters

The `Sakhi` model was trained from scratch with the following configuration:

```yaml
model_parameters:
  embed_dim: 2048
  num_heads: 8
  ff_dim: 4096
  chunk_length: 1024
  num_layers: 10
  vocab_size: 64000
```

- **Embedding Dimension**: 2048
- **Attention Heads**: 8
- **Feedforward Layer Dimension**: 4096 (with SwiGLU activation)
- **Context Length**: 1024 tokens
- **Layers**: 10 transformer decoder blocks
- **Vocabulary Size**: 64,000 (custom Byte-Level BPE)
- **Positional Encoding:** Rotary Positional Embeddings (RoPE), no learned or absolute positional embeddings

## Training Details

The model was pretrained for **100 hours** on **4× A100 GPUs** provided by **Lambda**. Pretraining was done using PyTorch with mixed precision and DDP (DistributedDataParallel) for efficient scaling.

```yaml
train_parameters:
  batch_size: 12
  num_epochs: 1
  init_learning_rate: 1e-5
  min_learning_rate: 1e-8
  seed: 42
  master_addr: "localhost"
  master_port: "12355"
  num_gpus: -1
  save_every_n_steps: 25000
  log_every_n_steps: 100
  gradient_clipping_max_norm: 3.0
  call_torch_compile_on_model: False
  gradient_accumulation_steps: 2
```

- **Effective Batch Size**: 12 × 2 (with gradient accumulation)
- **Epochs**: 1 (large-scale corpus, 13 billion tokens)
- **Learning Rate Schedule**: Linear warm-up to 1e-5, cosine decay to 1e-8
- **Gradient Clipping**: 3.0
- **Logging**: Every 100 steps using [Weights & Biases](https://wandb.ai/)
- **Checkpointing**: Every 25,000 steps

> 💡 Full Weights & Biases logs will be attached **(step x 100)** > [![Weights & Biases](https://img.shields.io/badge/Weights_%26_Biases-Project-blue)](https://api.wandb.ai/links/abhi11nav/g9oatq0u)

### Hardware Setup

- **GPUs**: 4 × A100 (Lambda)
- **Runtime**: 100 hours
- **Precision**: Mixed precision (FP16)

> 🚀 GPU costs were **partially sponsored by [Lambda Labs](https://lambdalabs.com/)**.

## Paths in configuration

```yaml
paths:
  tokenizer_path: "/"
  dataset_path: "/"
  save_dir: "/"
```

> ⚠️ Paths are placeholders — these should be replaced with actual paths

---

### 📚 References

For hyperparameter selection and RoPE implementation, the following tutorials were especially helpful:

- [📺 Stanford CS336](https://youtu.be/SQ3fZ1sAqXI?si=gymL4yFS60TkO8PY)
- [📺 Coding LLaMA 2 from scratch in PyTorch](https://youtu.be/oM4VmoabDAI?si=wfGdbLlfaJqJbR99)
