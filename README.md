<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/model_card.png" width="640"/>
</p>

<h1 align="center">🧠 BioLLama LLM</h1>
<h3 align="center">Fine tuned medical reasoning system using Llama 3 based architecture</h3>
<p align="center">
  <b>Developed by <a href="https://huggingface.co/calender">calender</a></b>  
  <br>
  <a href="https://huggingface.co/calender/BioLLama-LLM-Adapters">🔗 View model on Hugging Face</a>
</p>

---

### 🌿 Overview

BioLLama LLM is an end to end biomedical language model pipeline designed for medical question answering and reasoning.  
The adapter weights are hosted on **Hugging Face** for easy integration in downstream medical NLP systems.

---

### 🧩 Key Features

- Fine tuned via **LoRA** for lightweight domain specialization  
- Utilizes **4 bit quantized QLoRA** for efficient deployment  
- **Chain of thought** style reasoning for clinical interpretability  
- Trained on **MedMCQA** medical dataset  
- 72.7 percent performance on NEET PG 2024 subset

---

### ⚙️ Quick Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, PeftModel

base = "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025"
adapter = "calender/BioLLama-LLM-Adapters"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, adapter)

query = "Explain management of acute pulmonary embolism"
inputs = tokenizer(query, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
📊 Model Information
Field	Description
Base Model	ContactDoctor Bio Medical Llama 3 2 1B CoT 012025
Adapter	BioLLama LLM Adapters
Frameworks	Transformers, PEFT
License	Apache 2.0
Model Card	View on Hugging Face

📚 Citation
java
Copy code
@misc{calendar2025biollama,
  title = {BioLLama LLM},
  author = {Calendar, S.},
  year = {2025},
  publisher = {GitHub},
  note = {https://github.com/jikaan/BioLLama-LLM}
}
<p align="center"> 🧠 For complete weights and inference setup visit <a href="https://huggingface.co/calender/BioLLama-LLM-Adapters">BioLLama LLM Adapters on Hugging Face</a> </p> ```