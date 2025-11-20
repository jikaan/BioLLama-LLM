# BioLLama LLM: Medical Reasoning Adapter

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-View_Model-yellow)](https://huggingface.co/calender/BioLLama-LLM-Adapters)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)](https://github.com/huggingface/peft)

## Abstract

**BioLLama LLM** is a specialized biomedical language model designed for high-accuracy medical question answering and clinical reasoning. Built upon the **Llama 3.2 1B** architecture, this model utilizes Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) to achieve domain adaptation while maintaining computational efficiency.

The model is fine-tuned on the **MedMCQA** dataset and optimized for Chain-of-Thought (CoT) reasoning, making it suitable for downstream clinical support tasks and educational assessment.

## Performance Benchmark

| Metric | Score | Context |
| :--- | :--- | :--- |
| **NEET PG 2024 (Subset)** | **72.7%** | Zero-shot accuracy on select medical entrance examination questions. |

## Technical Architecture

This project implements a **QLoRA (Quantized LoRA)** approach to fine-tune a quantized base model. This allows for:
* **Memory Efficiency:** utilizing 4-bit quantization for deployment on consumer-grade hardware.
* **Modular Integration:** The adapter weights are decoupled from the base model, allowing for easy integration into existing Transformers pipelines.
* **Explainability:** The model is specifically trained to output reasoning steps before arriving at a diagnosis or answer.

## Model Specification

| Component | Details |
| :--- | :--- |
| **Base Model** | `ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025` |
| **Adapter Path** | `calender/BioLLama-LLM-Adapters` |
| **Architecture** | Llama 3.2 (1B Parameters) |
| **Quantization** | 4-bit (QLoRA) |
| **Training Data** | MedMCQA |
| **Frameworks** | PyTorch, Transformers, PEFT, BitsAndBytes |

## Inference

To utilize the model, ensure you have `transformers`, `peft`, and `torch` installed.

### Prerequisites

```bash
pip install torch transformers peft bitsandbytes accelerate

Python Implementation

The following script demonstrates how to load the base model with the BioLLama adapter for inference:
Python

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
BASE_MODEL_ID = "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025"
ADAPTER_ID = "calender/BioLLama-LLM-Adapters"

def load_model():
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # 2. Load Base Model (suggested: load in 4-bit or 8-bit for memory efficiency if needed)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 3. Apply LoRA Adapter
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    return model, tokenizer

def generate_response(query, model, tokenizer):
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Execution
if __name__ == "__main__":
    model, tokenizer = load_model()
    
    medical_query = "Explain the management of acute pulmonary embolism."
    response = generate_response(medical_query, model, tokenizer)
    
    print("-" * 50)
    print(f"Query: {medical_query}")
    print(f"Response:\n{response}")
    print("-" * 50)
```
Citation

If you use this model in your research or application, please cite it as follows:
Code snippet

@misc{calendar2025biollama,
  title = {BioLLama LLM: Fine-tuned Medical Reasoning System},
  author = {Calendar, S.},
  year = {2025},
  publisher = {Hugging Face},
  url = {[https://huggingface.co/calender/BioLLama-LLM-Adapters](https://huggingface.co/calender/BioLLama-LLM-Adapters)}
}

License

This project is licensed under the Apache 2.0 License.


---

