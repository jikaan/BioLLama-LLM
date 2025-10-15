<div align="center">

<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="90"/>

<h1 style="font-family: 'Segoe UI'; font-weight:700;">🧬 Med Llama SFT</h1>
<h3 style="font-family: 'Segoe UI'; font-weight:500;">Biomedical reasoning model with LoRA adapters and supervised chain of thought fine tuning</h3>

</div>

---

### 📘 Project Summary

This repository contains the training scripts, adapter weights, and environment configuration for **BioLLama SFT**, a fine tuned **medical question answering system** derived from **Llama 3 2 1B**.  
The model improves biomedical reasoning quality while maintaining small scale efficiency.

---

### 🧩 Repository Structure

project_root/
│
├── adapters/ ← LoRA adapter safetensors
├── examples/ ← Example inference snippets
├── requirements.txt ← Dependency list
├── single.txt ← Example fine tuning prompt
├── convo.txt ← Original supervised conversation sample
├── LICENSE ← Apache 2.0
└── README.md ← This documentation

yaml
Copy code

---

### ⚙️ Environment Setup

```bash
pip install -r requirements.txt
Dependencies include:

transformers

peft

bitsandbytes

torch

accelerate

🚀 Quick Inference Example
python
Copy code
from transformers import AutoModelForCausalLM, AutoTokenizer, PeftModel

base_model = "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025"
adapter_model = "./adapters"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, adapter_model)

query = "Describe the first line management of diabetic ketoacidosis"
inputs = tokenizer(query, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
📊 Results Summary
Dataset	Accuracy	Observation
MedMCQA	40 percent	Validation accuracy
NEET PG Clinical Subset	72.7 percent	Peak accuracy achieved

💡 Key Insights
LoRA adaptation reduced GPU memory by over 70 percent compared to full fine tuning

Chain of thought supervision improved interpretability

Quantized training stabilized gradient flow for small clinical datasets

🧠 Research Reference
Pal et al., MedMCQA: A Biomedical QA Dataset for Doctor Level Assessment, arXiv:2203.14371
Hu et al., LoRA: Low Rank Adaptation for Efficient Fine Tuning of Large Language Models, arXiv:2106.09685

📜 License
Released under Apache 2.0 License.
Free for research and educational applications.

<div align="center" style="font-family:'Segoe UI'; font-size:16px; font-weight:500;">
🧬 Developed by <b>Calendar S.</b>
🌿 Empowering medical research through responsible AI
⭐ Star the repository if you find it useful

</div>