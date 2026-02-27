# LLMOPs: Large Language Model Operations Pipeline

A comprehensive **MLOps framework** for fine-tuning and deploying Large Language Models (LLMs) using **Kubeflow Pipelines**, supporting multiple training strategies (**SFT, KD, DPO**) across **Code Generation** and **Intelligent Document Processing (IDP)** tasks.

---

## 🎯 Overview

LLMOPs provides an end-to-end workflow for:

- **Training Methods**
  - Supervised Fine-Tuning (SFT)
  - Knowledge Distillation (KD)
  - Direct Preference Optimization (DPO)

- **Supported Tasks**
  - Code Generation
  - Intelligent Document Processing (IDP)

- **Infrastructure**
  - Kubernetes-native
  - Built on Kubeflow Pipelines

- **Optimization**
  - QLoRA for training on 8GB GPUs

- **Experiment Tracking**
  - MLflow-enabled

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubeflow Pipelines                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   SFT       │  │     KD      │  │        DPO          │  │
│  │  Training   │  │  Training   │  │     Training        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │               │                    │             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Model     │  │   Model     │  │      Model          │  │
│  │ Evaluation  │  │ Evaluation  │  │   Evaluation        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │               │                    │             │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Model Deployment & Inference                │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```


---

## 🚀 Quick Start

### Prerequisites

- Kubernetes ≥ 1.20
- Kubeflow Pipelines installed
- NVIDIA GPU nodes
- Python 3.8+
- Docker
- Optional: MLflow server

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Narwal-AI-CoE/finetuning-distillation.git
cd finetuning-distillation
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Configure MLflow (optional)
```
export MLFLOW_TRACKING_URI=http://mlflow-service.mlflow.svc.cluster.local:5000
```


## ⚙️ Configuration

### Pipeline Parameters

```

| Parameter       | Type   | Description             | Example                    |
| --------------- | ------ | ----------------------- | -------------------------- |
| `student`       | string | Base model to fine-tune | `Qwen/Qwen2.5-3B-Instruct` |
| `data`          | string | Training dataset path   | `s3://bucket/data.jsonl`   |
| `save_dir`      | string | Output directory        | `/mnt/model`               |
| `epochs`        | int    | Training epochs         | `3`                        |
| `flow_type`     | string | Task type               | `IDP`                      |
| `train_type`    | string | Training method         | `SFT`                      |
| `schema`        | string | JSON schema path (IDP)  | `s3://bucket/schema.json`  |
| `document`      | string | Input document          | `"Sample document"`        |
| `eval_data`     | string | Evaluation dataset      | `s3://bucket/eval.json`    |
| `val_per_class` | int    | Samples per class       | `10`                       |
```

## ▶️ Running the Pipeline

### Compile & Submit Pipeline

```
python kfp_pipeline.py \
    --student Qwen/Qwen2.5-3B-Instruct \
    --data s3://your-bucket/training-data.jsonl \
    --flow_type IDP \
    --train_type SFT \
    --epochs 3 \
    --eval_data s3://your-bucket/eval-data.json \
    --val_per_class 20
```


### 📈 Monitoring & Tracking

#### MLflow Logging

   The pipeline logs:

- Hyperparameters

- Metrics (loss, accuracy, F1)

- Model artifacts

- GPU/resource usage

#### Kubeflow UI

Navigate to Pipelines → Finetune Pipeline to view:

- Run graphs

- Logs

- Artifacts

- Metrics


### 🔍 Example Workflows

#### 🧾 IDP with SFT

```
python kfp_pipeline.py \
    --student Qwen/Qwen2.5-3B-Instruct \
    --data s3://datasets/idp-sft.jsonl \
    --flow_type IDP \
    --train_type SFT \
    --epochs 3 \
    --schema s3://schemas/invoice-schema.json \
    --document "Invoice #12345 dated 2023-05-15..." \
    --eval_data s3://datasets/idp-eval.json \
    --val_per_class 25
```

#### 💻 Code Generation with DPO

```
python kfp_pipeline.py \
    --student Qwen/Qwen2.5-Coder-3B \
    --data s3://datasets/codegen-dpo.jsonl \
    --flow_type CodeGen \
    --train_type DPO \
    --epochs 2 \
    --prompt "Implement a binary search tree in Python" \
    --eval_data s3://datasets/codegen-eval.json \
    --val_per_class 20
```

#### 🔥 Knowledge Distillation

```
python kfp_pipeline.py \
    --student Qwen/Qwen2.5-3B-Instruct \
    --data s3://datasets/kd-data.jsonl \
    --flow_type IDP \
    --train_type KD \
    --epochs 4 \
    --use_logit_kd \
    --eval_data s3://datasets/idp-eval.json \
    --val_per_class 30
```

### 🛠️ Troubleshooting

#### GPU Memory Issues

- Reduce per_device_train_batch_size

- Increase gradient_accumulation_steps

- Enable 4-bit QLoRA

#### PVC Mount Issues

- Verify StorageClass

- Check PVC access modes

- Ensure capacity quota

#### MLflow Errors

- Check MLFLOW_TRACKING_URI

- Test service connectivity

- Validate authentication

#### Slow Training

- Enable gradient checkpointing

- Use Flash Attention (if supported)

- Improve dataset caching


### Debug Mode

```
export KFP_LOG_LEVEL=DEBUG
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 📚 Advanced Usage

### Custom Component Development

- Create a folder under `ops/`
- Write a `Dockerfile` for the component
- Create the component YAML file
- Import it inside `kfp_pipeline.py`

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repository  
2. Create a feature branch  
3. Make your changes  
4. Add tests  
5. Open a Pull Request  

---



## 🙏 Acknowledgments

- **Kubeflow** — for orchestration  
- **Hugging Face** — for model ecosystem  
- **MLflow** — for experiment tracking  
- **PEFT** — for efficient fine-tuning  
