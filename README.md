#  A Dual-Layer Framework for Detecting and Detoxifying Toxic language




This repository contains the official code for my MSc thesis on **toxic language detection and detoxification**, using a two-stage approach:

1. **Stage 1 – Toxicity Classification:**  
   A ModernBERT-based model classifies toxic content using binary and fine-grained multi-label classification.

2. **Stage 2 – Detoxification via Prompting and Fine-tuning:**  
   Detoxification is performed using prompting or fine-tuning with large language models (LLama-3-8B, Mistral-7B and mT0), with and without the Stage 1 output in the detox prompt.

---

## 📁 Repository Structure

```bash
my-thesis-code/
├── notebooks/               # Final project notebooks (pipeline, training, evaluation)
│   ├── 01_load_dataset.ipynb
│   ├── 02_classification_pipeline.ipynb
│   ├── 03_stage1_classifier_inference.ipynb
│   ├── 04_llama_prompting_without_stage1.ipynb
│   ├── 05_llama_prompting_with_stage1.ipynb
│   ├── 06_Mistral_7B_Instruct_v0_2_detox_prompt_without_stage_one.ipynb
│   ├── 07_Mistral_7B_Instruct_v0_2_detox_prompt_with_stage_1.ipynb
│   ├── 08_train_and_eval_mt0base_without_stage1.ipynb
│   ├── 09_train_and_eval_mt0base_with_stage1.ipynb
│   └── 05_stage1_evaluation_hamming.ipynb
├── experiments/             # Archived experimental training runs
│   ├── train_moderbert_classifier_160k_example.ipynb
│   ├── training_mt0_base_on_multilingual_paradetox.ipynb
│   └── training_mt0_base_on_multilingual_paradetox_with_stage1.ipynb
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignore patterns for Git
└── README.md                # Project documentation
```

---

## 🧠 Project Overview

This project proposes a modular, explainable pipeline for detoxifying toxic language:

- A **ModernBERT-based classifier** first identifies toxic content and categorizes it across binary and multiple labels.
- A second stage performs **detoxification**, tested with:
  - Prompt-only methods (Mistral-7B, LLaMA-3-8B)
  - Fine-tuned models (mT0-base) with and without Stage 1 context

The system was evaluated using **PAN 2024/2025 detoxification metrics** and Hamming loss for multi-label classification.

---

## 🧪 Datasets Used

This project uses the following datasets:

- **[Jigsaw Toxic Comment Classification (Kaggle)](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)**  

- **[ParadeTox (Hugging Face)](https://huggingface.co/datasets/s-nlp/paradetox)**  
 

- **[Multilingual ParadeTox (textdetox)](https://huggingface.co/datasets/textdetox/multilingual_paradetox)**  



> 📂 Datasets are **not included** in this repo.  



---

## ▶️ How to Run Each Stage

### 🧩 Stage 1: Toxicity Classification

- Train ModernBERT classifier:
  - See `experiments/train_moderbert_classifier_160k_example.ipynb`
- Use classifier pipeline:
  - See `notebooks/02_classification_pipeline.ipynb`
- Inference-only version:
  - See `03_stage1_classifier_inference.ipynb`
- Evaluation with Hamming loss:
  - See `05_stage1_evaluation_hamming.ipynb`

### 🧹 Stage 2: Detoxification

#### Prompt-based:
- LLaMA detox without classification: `04_llama_prompting_without_stage1.ipynb`
- LLaMA detox with classification: `05_llama_prompting_with_stage1.ipynb`
- Mistral-7B detox without classification: `06_..._without_stage_one.ipynb`
- Mistral-7B detox with classification: `07_..._with_stage_1.ipynb`

#### Fine-tuned:
- `08_train_and_eval_mt0base_without_stage1.ipynb`
- `09_train_and_eval_mt0base_with_stage1.ipynb`

---

## 📊 Evaluation

This project uses official **PAN detoxification metrics**:
- Sentence Transformer Alignment (STA)
- Semantic Similarity
- CHRF1
- J-Score

Classification metrics:

- Precision, Recall, F1-score and Macro F1-score 
- **Hamming loss**

---

## 🙋 Contact

For questions or contributions, feel free to open an issue or contact me at:
- 📧 mahrokhhassani99@gmail.com
