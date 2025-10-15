import random
import logging
import os
import re
import torch
import gc

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback,
    GenerationConfig
)
from datasets import load_dataset, DatasetDict
from trl import SFTTrainer

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# --- Configuration ---
# Model Configuration
BASE_MODEL_NAME = "your_model_name_here"  # Replace with your model name
DATASET_NAME = "your_dataset_name_here"    # Replace with your dataset name
DATASET_SUBSET_PERCENT = 0.8  # Adjust as needed

# Paths - Replace with your local paths
CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./output"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- PEFT LoRA Config ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# --- Training Hyperparameters ---
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-5
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
FP16_ENABLED = True
GRADIENT_CHECKPOINTING = True

# --- Resource Management & Sequence Length ---
FILTER_TOKEN_THRESHOLD = 4000 # Filter examples likely exceeding model limits after formatting
logger.info(f"Examples with estimated raw token count > {FILTER_TOKEN_THRESHOLD} will be REMOVED.")

# --- Training Steps ---
LOGGING_STEPS = 200
EVAL_STEPS = 800
SAVE_STEPS = 800
SAVE_TOTAL_LIMIT = 2

# --- Metrics Evaluation ---
EVAL_SUBSET_SIZE = 100
GENERATION_MAX_LENGTH = 512

# --- Global variables ---
EOS_TOKEN = None; processed_data = None; tokenizer = None; model = None; trainer = None
lora_config = None
current_eval_dataset_for_metrics = None # For compute_metrics callback

# --- Model and Tokenizer Loading ---
logger.info("--- Loading Model and Tokenizer ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    logger.info("Tokenizer loaded.")
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token; logger.info("Set pad_token to eos_token.")
        else:
            raise ValueError("EOS token missing, cannot set pad_token.")
    if tokenizer.eos_token is None: raise ValueError("EOS token is missing in the tokenizer.")
    EOS_TOKEN = tokenizer.eos_token; logger.info(f"Using EOS: {EOS_TOKEN}, PAD: {tokenizer.pad_token}")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    logger.info("BitsAndBytesConfig created (compute dtype: float16).")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    logger.info("Base model loaded.")

    if model.config.pad_token_id != tokenizer.pad_token_id:
         model.config.pad_token_id = tokenizer.pad_token_id
         logger.info("Synced model pad_token_id with tokenizer.")

except Exception as e:
     logger.error(f"Failed model/tokenizer load: {e}", exc_info=True); model = None; tokenizer = None

if model:
    logger.info("--- Setting up PEFT ---")
    try:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        logger.info("Model prepared for k-bit training.")
        lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES, lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM)
        logger.info("LoRA config created.")
        model = get_peft_model(model, lora_config)
        logger.info("PEFT LoRA model created."); model.print_trainable_parameters()
    except Exception as e: logger.error(f"PEFT setup failed: {e}", exc_info=True)
else: logger.warning("Skipping PEFT setup: Model not loaded.")


# --- Data Validation and Formatting Functions ---
def validate_example(example):
    required_fields = ['question', 'opa', 'opb', 'opc', 'opd', 'cop']
    if not all(field in example and example[field] is not None for field in required_fields): return False
    for field in ['question', 'opa', 'opb', 'opc', 'opd']:
        if not isinstance(example.get(field), str) or not example.get(field).strip(): return False
    cop_val = example['cop']; idx = -1
    try:
        if isinstance(cop_val, str) and cop_val.strip().isdigit():
            val = int(cop_val.strip()); idx = val - 1 if 1 <= val <= 4 else (val if 0 <= val <= 3 else -1)
        elif isinstance(cop_val, int):
            idx = cop_val - 1 if 1 <= cop_val <= 4 else (cop_val if 0 <= cop_val <= 3 else -1)
        else: return False
        if not (0 <= idx <= 3): return False
    except (ValueError, TypeError): return False
    return True

def parse_correct_option(cop_value):
    idx = -1
    try:
        if isinstance(cop_value, str) and cop_value.strip().isdigit():
            val = int(cop_value.strip()); idx = val - 1 if 1 <= val <= 4 else (val if 0 <= val <= 3 else -1)
        elif isinstance(cop_value, int):
            idx = cop_value - 1 if 1 <= cop_value <= 4 else (cop_value if 0 <= cop_value <= 3 else -1)
        if 0 <= idx <= 3:
            mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}; return idx, mapping[idx]
    except Exception: pass
    return None, None

def format_cot_for_sft(example, tokenizer_eos_token):
    """Formats the example and ensures truncation."""
    if not tokenizer_eos_token or not isinstance(tokenizer_eos_token, str): return None
    index, correct_letter = parse_correct_option(example.get('cop'))
    if correct_letter is None or index is None: return None
    question = example.get('question', '').strip()
    opa = example.get('opa', '').strip(); opb = example.get('opb', '').strip()
    opc = example.get('opc', '').strip(); opd = example.get('opd', '').strip()
    if not all([question, opa, opb, opc, opd]): return None
    subject_name = example.get('subject_name', '').strip()
    topic_name = example.get('topic_name', '').strip()
    context_header = ""
    if subject_name: context_header += f"Subject: {subject_name}\n"
    if topic_name: context_header += f"Topic: {topic_name}\n"
    if context_header: context_header += "\n"
    explanation_raw = example.get('exp')
    explanation = explanation_raw.strip() if explanation_raw is not None else ""
    if explanation:
        explanation_clean = re.sub(r'^\s*(Ans|Answer)\s*[:.\-]?\s*[A-Da-d]\b\s*\.?\s*', '', explanation, flags=re.IGNORECASE | re.MULTILINE).strip()
        explanation_text = f"Comprehensive Explanation: {explanation_clean}"
    else: explanation_text = "Explanation: No detailed explanation provided in the dataset."
    options_list = [(0, opa), (1, opb), (2, opc), (3, opd)]
    random.shuffle(options_list)
    new_answer_idx = -1; option_texts_shuffled = []
    for i, (original_index, text) in enumerate(options_list):
        option_texts_shuffled.append(f"{chr(65+i)}: {text}")
        if original_index == index: new_answer_idx = i
    option_text = "\n".join(option_texts_shuffled)
    if new_answer_idx == -1: return None
    full_prompt = (
        f"{context_header}"
        f"**Medical Reasoning Task**\nQuestion: {question}\n\nOptions:\n{option_text}\n\n"
        f"**Required Analysis Format:**\n1. Identify key clinical factors presented in the question.\n"
        f"2. Systematically analyze each option based on the clinical factors and underlying medical knowledge.\n"
        f"3. Compare the pathophysiology or mechanisms relevant to the options.\n"
        f"Final Answer: The correct option is {chr(65 + new_answer_idx)}.\n{explanation_text}{tokenizer_eos_token}"
    )
    # Enforce truncation during tokenization
    truncated_tokens = tokenizer.encode(full_prompt, truncation=True, max_length=4096, add_special_tokens=False)
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    result = {
        "text": truncated_text,  # Use the truncated text
        "original_cop_index": index,
        "original_cop_letter": correct_letter,
        "shuffled_cop_index": new_answer_idx,
        "shuffled_cop_letter": chr(65 + new_answer_idx),
        "subject_name": subject_name,
        "topic_name": topic_name
    }
    return result

def count_tokens(example):
    """Estimates token count based on individual fields. Handles errors gracefully."""
    try:
        q_tokens = tokenizer.encode(str(example.get("question", "")), add_special_tokens=False)
        opa_tokens = tokenizer.encode(str(example.get("opa", "")), add_special_tokens=False)
        opb_tokens = tokenizer.encode(str(example.get("opb", "")), add_special_tokens=False)
        opc_tokens = tokenizer.encode(str(example.get("opc", "")), add_special_tokens=False)
        opd_tokens = tokenizer.encode(str(example.get("opd", "")), add_special_tokens=False)
        exp_tokens = tokenizer.encode(str(example.get("exp", "")), add_special_tokens=False)
        cop_tokens = tokenizer.encode(str(example.get("cop", "")), add_special_tokens=False)
        subject_tokens = tokenizer.encode(str(example.get("subject_name", "")), add_special_tokens=False)
        topic_tokens = tokenizer.encode(str(example.get("topic_name", "")), add_special_tokens=False)

        total = (len(q_tokens) + len(opa_tokens) + len(opb_tokens) +
                 len(opc_tokens) + len(opd_tokens) + len(exp_tokens) +
                 len(cop_tokens) + len(subject_tokens) + len(topic_tokens))

        return {"total_tokens": total}
    except Exception as e:
        logger.warning(f"Error token counting example {example.get('id', 'N/A')}: {e}")
        return {"total_tokens": float('inf')}  # Mark as exceeding the limit

if tokenizer and EOS_TOKEN:
    logger.warning("Dataset loading section needs to be implemented for your specific dataset.")
    processed_data = None
else: logger.warning("Skipping data processing: Tokenizer or EOS_TOKEN missing.")


def extract_answer_letter(generated_text):
    pattern = r"Final Answer: The correct option is\s*([A-D])\b"
    matches = re.findall(pattern, generated_text, re.IGNORECASE | re.MULTILINE)
    if matches: return matches[-1].upper()
    pattern_alt = r"correct option is\s*([A-D])\b"
    matches_alt = re.findall(pattern_alt, generated_text, re.IGNORECASE | re.MULTILINE)
    if matches_alt: return matches_alt[-1].upper()
    return None

def compute_metrics(eval_pred):
    global model, tokenizer, current_eval_dataset_for_metrics
    evaluation_data = current_eval_dataset_for_metrics
    if model is None or tokenizer is None or evaluation_data is None:
        logger.error("compute_metrics missing model, tokenizer, or eval dataset ref.")
        return {"answer_accuracy": 0.0}
    if len(evaluation_data) == 0:
        logger.warning("Evaluation dataset for metrics is empty.")
        return {"answer_accuracy": 0.0, "eval_subset_size": 0, "correct_count": 0}
    actual_eval_size = min(len(evaluation_data), EVAL_SUBSET_SIZE if EVAL_SUBSET_SIZE > 0 else len(evaluation_data))
    if actual_eval_size <= 0:
         logger.warning(f"Eval subset size is {actual_eval_size}. Skipping metrics.")
         return {"answer_accuracy": 0.0, "eval_subset_size": 0, "correct_count": 0}
    logger.info(f"Starting custom metrics generation for {actual_eval_size} examples...")
    subset_indices = random.sample(range(len(evaluation_data)), actual_eval_size)
    eval_subset = evaluation_data.select(subset_indices)
    gen_pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_config = GenerationConfig(max_new_tokens=GENERATION_MAX_LENGTH, do_sample=False, pad_token_id=gen_pad_token_id, eos_token_id=tokenizer.eos_token_id)
    results = []; correct_count = 0; model.eval()
    prompt_marker = "**Required Analysis Format:**"
    for i, example in enumerate(eval_subset):
        if not all(k in example for k in ['text', 'shuffled_cop_letter']):
            logger.warning(f"Skipping metrics ex {i}: Missing 'text' or 'shuffled_cop_letter'."); results.append({"correct": False, "error": "Missing keys"}); continue
        input_text = example['text']; correct_letter = example['shuffled_cop_letter']
        prompt_part = input_text; search_pattern = f"Final Answer: The correct option is {correct_letter}."; idx_final_answer = input_text.find(search_pattern)
        if idx_final_answer != -1: prompt_part = input_text[:idx_final_answer].strip()
        else:
             idx_marker = input_text.find(prompt_marker)
             if idx_marker != -1: prompt_part = input_text[:idx_marker + len(prompt_marker)].strip()
             else:
                  parts = input_text.split("Final Answer:", 1)
                  if len(parts) > 0: prompt_part = parts[0].strip()
                  else: logger.warning(f"Skipping metrics ex {i}: Cannot extract prompt."); results.append({"correct": False, "error": "Prompt extraction failed"}); continue
        if not prompt_part: logger.warning(f"Skipping metrics ex {i}: Extracted prompt part is empty."); results.append({"correct": False, "error": "Empty prompt"}); continue
        inputs = tokenizer(
            prompt_part,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=False
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=gen_config)
                input_token_len = inputs['input_ids'].shape[1]; generated_tokens = outputs[0][input_token_len:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                predicted_letter = extract_answer_letter(generated_text)
                is_correct = bool(predicted_letter and correct_letter and predicted_letter == correct_letter)
                if is_correct: correct_count += 1
                results.append({"correct": is_correct, "predicted": predicted_letter if predicted_letter else "Extraction Failed", "ground_truth": correct_letter})
        except Exception as e: logger.error(f"Generation error on metrics ex {i}: {e}", exc_info=False); results.append({"correct": False, "error": str(e)})
        if (i + 1) % 50 == 0: logger.info(f"  Custom metrics progress: {i+1}/{actual_eval_size}")
    total_valid_results = len([r for r in results if "error" not in r])
    accuracy = correct_count / total_valid_results if total_valid_results > 0 else 0.0
    prediction_stats = {};
    for r in results: pred = r.get("predicted", "Error/Skipped"); prediction_stats[pred] = prediction_stats.get(pred, 0) + 1
    metrics = {"answer_accuracy": accuracy, "eval_subset_size": actual_eval_size, "eval_processed_count": len(results), "eval_valid_count": total_valid_results, "correct_count": correct_count, "prediction_distribution": prediction_stats}
    logger.info(f"Custom Eval Metrics Calculation Complete: {metrics}")
    return metrics

class AnswerAccuracyCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        try:
            if 'trainer' in kwargs:
                trainer_instance = kwargs['trainer']
                if current_eval_dataset_for_metrics is None: logger.warning("current_eval_dataset_for_metrics is None in callback.")
                custom_metrics = compute_metrics(None)
                if state is not None and custom_metrics and "answer_accuracy" in custom_metrics:
                    log_metrics = {"eval_answer_accuracy": custom_metrics["answer_accuracy"]}
                    trainer_instance.log(log_metrics)
            else: logger.warning("Trainer instance not found in callback kwargs.")
        except Exception as e: logger.error(f"Callback error: {e}", exc_info=True)


logger.info("--- Configuring Trainer ---")
training_args = None; trainer = None
if model and tokenizer and lora_config and processed_data and 'train' in processed_data and len(processed_data['train']) > 0:
    try:
        logger.info("Creating Training Arguments...")
        if CHECKPOINT_DIR: os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=CHECKPOINT_DIR,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, learning_rate=LEARNING_RATE, lr_scheduler_type=LR_SCHEDULER_TYPE,
            warmup_ratio=WARMUP_RATIO, weight_decay=WEIGHT_DECAY, optim="paged_adamw_8bit", fp16=FP16_ENABLED, bf16=False,
            gradient_checkpointing=GRADIENT_CHECKPOINTING, gradient_checkpointing_kwargs={'use_reentrant': False}, max_grad_norm=1.0,
            logging_dir=os.path.join(CHECKPOINT_DIR, 'logs') if CHECKPOINT_DIR else None,
            logging_strategy="steps", logging_steps=LOGGING_STEPS,
            eval_strategy="steps", eval_steps=EVAL_STEPS,
            save_strategy="steps", save_steps=SAVE_STEPS, save_total_limit=SAVE_TOTAL_LIMIT,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],
            remove_unused_columns=False,
        )
        logger.info("Training Arguments configured.")

        logger.info("Determining evaluation dataset for Trainer...")
        eval_dataset_for_trainer = processed_data.get("validation")
        if eval_dataset_for_trainer is None or len(eval_dataset_for_trainer) == 0:
             logger.warning("Validation split missing or empty. Using test split for standard Trainer eval if available.")
             eval_dataset_for_trainer = processed_data.get("test")
             if eval_dataset_for_trainer is None or len(eval_dataset_for_trainer) == 0:
                  logger.warning("No validation or test split available for standard evaluation during training. Disabling Trainer eval.")
                  training_args.evaluation_strategy = "no"
                  training_args.load_best_model_at_end = False
                  eval_dataset_for_trainer = None

        logger.info("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_data["train"],
            eval_dataset=eval_dataset_for_trainer,
            peft_config=lora_config,
            formatting_func=lambda example: example.get('text', ''),
            callbacks=[AnswerAccuracyCallback()],
        )
        logger.info("SFTTrainer initialized.")

    except Exception as e:
        logger.error(f"Trainer setup failed: {e}", exc_info=True); trainer = None
else:
    logger.warning("Prerequisites not met for Trainer setup (model, tokenizer, data, etc.). Skipping.")

if trainer:
    logger.info("--- Starting Training with Metrics Calculation ---")
    logger.info("Clearing CUDA cache...")
    torch.cuda.empty_cache(); gc.collect()

    logger.info("Setting global dataset reference for potential mid-training evaluations...")
    current_eval_dataset_for_metrics = eval_dataset_for_trainer
    if current_eval_dataset_for_metrics:
        logger.info(f"Callback compute_metrics will use '{ 'validation' if eval_dataset_for_trainer == processed_data.get('validation') else 'test' }' split during training.")
    else:
        logger.warning("No eval dataset set for Trainer, callback compute_metrics will likely fail if triggered during training.")

    try:
        logger.info("Calling trainer.train()...")
        train_result = trainer.train()
        logger.info("Training completed successfully.")
        metrics = train_result.metrics

        # Calculate metrics after every 800 steps (simplified)
        if trainer.state.global_step % 800 == 0:
            logger.info(f"Training progress: Step {trainer.state.global_step} completed")

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", os.path.join(OUTPUT_DIR, "train_results.json"))

        trainer.save_state(os.path.join(OUTPUT_DIR, "trainer_state"))
        logger.info("Training metrics and final trainer state saved.")

        logger.info(f"Saving the final model adapter to {OUTPUT_DIR}...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        trainer.save_model(OUTPUT_DIR)
        if tokenizer:
            tokenizer.save_pretrained(OUTPUT_DIR)
            logger.info(f"Model adapter and tokenizer saved to {OUTPUT_DIR}")
        else: logger.warning("Tokenizer was None, cannot save tokenizer.")

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        logger.info("Attempting save after error...")
        try:
            state_path = os.path.join(OUTPUT_DIR, "trainer_state_on_error") if OUTPUT_DIR else "trainer_state_on_error"
            trainer.save_state(state_path); logger.info(f"State saved to {state_path}")
        except Exception as save_e: logger.error(f"State save failed: {save_e}")
        try:
            err_path = os.path.join(OUTPUT_DIR, "model_on_error") if OUTPUT_DIR else "model_on_error"
            os.makedirs(err_path, exist_ok=True)
            trainer.save_model(err_path)
            if tokenizer: tokenizer.save_pretrained(err_path)
            logger.info(f"Model/tokenizer saved after error to {err_path}")
        except Exception as sm_e: logger.error(f"Model save after error failed: {sm_e}")

else: logger.warning("Skipping training: Trainer not initialized.")


if trainer and processed_data:
    logger.info("--- Final Evaluation (using best model if loaded) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Clearing CUDA cache before final evaluation..."); torch.cuda.empty_cache(); gc.collect()

    logger.info("--- Evaluating on Validation Set ---")
    if "validation" in processed_data and len(processed_data["validation"]) > 0:
        try:
            current_eval_dataset_for_metrics = processed_data["validation"]
            logger.info(f"Set metrics dataset ref to validation split ({len(current_eval_dataset_for_metrics)} examples).")
            eval_results = trainer.evaluate(eval_dataset=processed_data["validation"])
            logger.info(f"Final Validation Results (standard): {eval_results}")
            trainer.log_metrics("eval_final_standard", eval_results)
            trainer.save_metrics("eval_final_standard", os.path.join(OUTPUT_DIR, "eval_results_validation_standard.json"))
            logger.info("Running custom metrics separately on final validation set...")
            final_val_custom_metrics = compute_metrics(None)
            trainer.log_metrics("eval_final_custom", final_val_custom_metrics)
            trainer.save_metrics("eval_final_custom", os.path.join(OUTPUT_DIR, "eval_results_validation_custom.json"))
        except Exception as e:
            logger.error(f"Final validation evaluation error: {e}", exc_info=True)
    else:
        logger.warning("No validation split found/is empty. Skipping final validation evaluation.")

    logger.info("--- Evaluating on Test Set ---")
    if "test" in processed_data and len(processed_data["test"]) > 0:
        try:
            current_eval_dataset_for_metrics = processed_data["test"]
            logger.info(f"Set metrics dataset ref to test split ({len(current_eval_dataset_for_metrics)} examples).")
            test_results = trainer.evaluate(processed_data["test"], metric_key_prefix="test")
            logger.info(f"Test results (standard): {test_results}")
            trainer.log_metrics("test_standard", test_results)
            trainer.save_metrics("test_standard", os.path.join(OUTPUT_DIR, "eval_results_test_standard.json"))
            logger.info("Running custom accuracy metrics separately on test set...")
            test_custom_metrics = compute_metrics(None)
            logger.info(f"Test Answer Accuracy Results: {test_custom_metrics}")
            trainer.log_metrics("test_custom", test_custom_metrics)
            trainer.save_metrics("test_custom", os.path.join(OUTPUT_DIR, "eval_results_test_custom.json"))
        except Exception as e:
            logger.error(f"Final test evaluation error: {e}", exc_info=True)
    else:
        logger.warning("No 'test' split found/is empty. Skipping final test evaluation.")

else:
    logger.warning("Skipping final eval: Trainer or processed_data not available.")

logger.info("--- Script Finished ---")