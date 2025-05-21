# stage3_active_train.py (Advanced Phased & Validated Active Learning Cycle - Memory Optimization)

import os
import json
import random
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm
import logging
import shutil 
import sys 

# --- Configuration ---
STAGE2_OUTPUT_POOL = "./active_learn_pool.jsonl"
BASE_MODEL_NAME = "Salesforce/codegen-350M-mono"
OUTPUT_DIR_BASE = "./active_learning_cycle_0_gpu_cpu" 
LOG_FILE = os.path.join(OUTPUT_DIR_BASE, "s3_active_train_cycle_0_gpu_cpu.log")

# --- Ensure Output Directories Exist ---
try:
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
except Exception as e:
    print(f"CRITICAL ERROR: Could not create output/log directories: {e}")
    
# --- Robust Logging Setup ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
try:
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - [%(levelname)s] - %(name)s - %(module)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'), 
            logging.StreamHandler(sys.stdout)      
        ]
    )
    logger = logging.getLogger(__name__) 
    logger.info("--- Logging configured successfully. Script starting. ---")
    print("--- PRINT: Logging configured. Script starting. ---") 
except Exception as e:
    print(f"CRITICAL ERROR: Failed to configure logging: {e}")
    class FallbackLogger:
        def info(self, msg): print(f"LOG_FALLBACK INFO: {msg}")
        def warning(self, msg): print(f"LOG_FALLBACK WARNING: {msg}")
        def error(self, msg, exc_info=False): print(f"LOG_FALLBACK ERROR: {msg}")
        def debug(self, msg): print(f"LOG_FALLBACK DEBUG: {msg}")
    logger = FallbackLogger()
    logger.error(f"Using fallback print-based logger due to basicConfig failure: {e}")

# --- Device Setup (GPU/CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# --- Phased Training Configuration ---
TRAINING_PHASES = [
    {
        "phase_name": "Python_Focus",
        "language_filter": "python", 
        "num_samples": 300, 
        "epochs": 3,
        "learning_rate": 3e-5,
        "warmup_ratio": 0.1,
        "freeze_layers_config": None, 
        "python_replay_ratio": 0.0 
    },
    {
        "phase_name": "CPP_Introduction",
        "language_filter": "cpp", 
        "num_samples": 200,
        "epochs": 3, 
        "learning_rate": 2e-5, 
        "warmup_ratio": 0.1,
        "freeze_layers_config": {"num_embedding_layers_to_freeze": 0, "num_transformer_blocks_to_freeze": 6, "unfreeze_after_epoch": 1},
        "python_replay_ratio": 0.15 
    },
    {
        "phase_name": "Java_Introduction",
        "language_filter": "java", 
        "num_samples": 200,
        "epochs": 3,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "freeze_layers_config": {"num_embedding_layers_to_freeze": 0, "num_transformer_blocks_to_freeze": 6, "unfreeze_after_epoch": 1},
        "python_replay_ratio": 0.10 
    },
    {
        "phase_name": "Final_Mixed_Tune",
        "language_filter": None, 
        "num_samples": 250, 
        "epochs": 2,
        "learning_rate": 1e-5, 
        "warmup_ratio": 0.1,
        "freeze_layers_config": None, 
        "python_replay_ratio": 0.20 
    }
]

# General Training Hyperparameters
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 1 # <<< REDUCED BATCH SIZE FOR MEMORY
VALIDATION_SPLIT_RATIO = 0.15
EARLY_STOPPING_PATIENCE = 3


# --- Layer Freezing Utilities ---
def freeze_model_layers(model, num_embedding_layers_to_freeze=0, num_transformer_blocks_to_freeze=0):
    logger.debug(f"Attempting to freeze layers: Embeddings={num_embedding_layers_to_freeze}, Transformers={num_transformer_blocks_to_freeze}")
    frozen_params_count = 0 
    if num_embedding_layers_to_freeze > 0 and hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        for param in model.transformer.wte.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_params_count += param.numel()
        if frozen_params_count > 0 : logger.info(f"Froze word token embedding layer (wte). Parameters affected: {frozen_params_count}")

    transformer_params_frozen_this_call = 0
    if num_transformer_blocks_to_freeze > 0 and hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        for i, block in enumerate(model.transformer.h):
            if i < num_transformer_blocks_to_freeze:
                block_frozen_params = 0
                for param in block.parameters():
                    if param.requires_grad: 
                        param.requires_grad = False
                        block_frozen_params += param.numel() 
                if block_frozen_params > 0:
                    logger.info(f"Froze transformer block {i}. Parameters affected: {block_frozen_params}")
                    transformer_params_frozen_this_call += block_frozen_params
    
    total_frozen_this_call = (frozen_params_count - transformer_params_frozen_this_call) + transformer_params_frozen_this_call 
    if total_frozen_this_call > 0: 
        logger.info(f"Total parameters newly frozen in this call: {total_frozen_this_call}")
    elif num_embedding_layers_to_freeze > 0 or num_transformer_blocks_to_freeze > 0:
        logger.info("No new parameters were frozen (or specified layers already frozen).")
    return model

def unfreeze_all_layers(model):
    unfrozen_params_count = 0
    for name, param in model.named_parameters(): 
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen_params_count += param.numel()
    if unfrozen_params_count > 0:
        logger.info(f"Unfroze {unfrozen_params_count} parameters. All model layers are now trainable.")
    else:
        logger.info("All model layers were already trainable.")
    return model

# --- Custom Dataset for Fine-tuning ---
class FineTuningDataset(TorchDataset):
    def __init__(self, data_records, tokenizer, max_length):
        self.records = data_records
        self.tokenizer = tokenizer
        self.max_length = max_length
        if not self.records:
            logger.warning("FineTuningDataset initialized with zero records.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        text = record.get("raw_content", "")
        if not text: 
            logger.warning(f"Record at index {idx} (ID: {record.get('snippet_id')}) has empty raw_content. Using EOS token as fallback.")
            text = self.tokenizer.eos_token if self.tokenizer.eos_token else "<|endoftext|>" 

        try:
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length", 
                truncation=True,
                return_tensors="pt" 
            )
            return {
                "input_ids": inputs.input_ids.squeeze(0),
                "attention_mask": inputs.attention_mask.squeeze(0)
            }
        except Exception as e:
            logger.error(f"Error tokenizing record at index {idx} (ID: {record.get('snippet_id')}): {e}. Content snippet: {text[:100]}", exc_info=True)
            dummy_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            return {
                "input_ids": torch.tensor([dummy_id] * self.max_length, dtype=torch.long),
                "attention_mask": torch.tensor([0] * self.max_length, dtype=torch.long) 
            }


# --- Helper to Load Data Pool ---
def load_data_pool(pool_file_path):
    logger.info(f"Attempting to load data pool from: {pool_file_path}")
    if not os.path.exists(pool_file_path):
        logger.error(f"Data pool file not found: {pool_file_path}")
        return []
    
    pool_data = []
    try:
        with open(pool_file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    pool_data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON line {i+1} in {pool_file_path}: {line.strip()[:100]}")
        logger.info(f"Loaded {len(pool_data)} records from {pool_file_path}")
    except Exception as e:
        logger.error(f"Failed to read or parse data pool file {pool_file_path}: {e}", exc_info=True)
        return [] 
    return pool_data

# --- Data Selection Strategy ---
def select_data_for_phase(data_pool, num_main_samples, language_filter, 
                          python_replay_ratio=0.0, num_python_replay_samples=0, 
                          previously_selected_ids=None):
    logger.debug(f"select_data_for_phase called. num_main_samples: {num_main_samples}, language_filter: {language_filter}, replay_ratio: {python_replay_ratio}")
    if not data_pool:
        logger.warning("Data pool is empty for selection.")
        return [], set()

    if previously_selected_ids is None:
        previously_selected_ids = set()
    logger.debug(f"Previously selected IDs count: {len(previously_selected_ids)}")


    available_samples = [r for r in data_pool if r.get("snippet_id") not in previously_selected_ids]
    logger.debug(f"Available samples after excluding previously selected: {len(available_samples)}")
    if not available_samples:
        logger.warning("No available samples left in the pool after excluding previously selected ones.")
        return [], previously_selected_ids 

    selected_main_samples = []
    selected_replay_samples = []

    if language_filter: 
        main_lang_pool = [r for r in available_samples if r.get("language") == language_filter] 
        logger.debug(f"Pool for main language '{language_filter}': {len(main_lang_pool)} samples.")
        if not main_lang_pool:
            logger.warning(f"No samples found for main language '{language_filter}' from available pool.") 
        else:
            num_to_select = min(num_main_samples, len(main_lang_pool))
            selected_main_samples = random.sample(main_lang_pool, num_to_select)
            logger.info(f"Selected {len(selected_main_samples)} samples for main language: {language_filter}") 
    elif num_main_samples > 0 : 
        num_to_select = min(num_main_samples, len(available_samples))
        selected_main_samples = random.sample(available_samples, num_to_select)
        logger.info(f"Selected {len(selected_main_samples)} samples from all available languages (no specific main filter).")

    current_phase_selected_ids = {s.get("snippet_id") for s in selected_main_samples if s.get("snippet_id")}
    
    actual_python_replay_samples_to_select = 0
    if python_replay_ratio > 0 and language_filter != "python": 
        actual_python_replay_samples_to_select = int(len(selected_main_samples) * python_replay_ratio)
    elif num_python_replay_samples > 0 and language_filter != "python": 
        actual_python_replay_samples_to_select = num_python_replay_samples

    if actual_python_replay_samples_to_select > 0:
        replay_candidate_pool = [
            r for r in data_pool 
            if r.get("language") == "python" and \
               r.get("snippet_id") not in current_phase_selected_ids and \
               r.get("snippet_id") not in previously_selected_ids
        ]
        logger.debug(f"Python replay candidate pool size: {len(replay_candidate_pool)}")
        if not replay_candidate_pool:
            logger.warning("No Python samples available for replay that haven't been selected.")
        else:
            num_to_select_replay = min(actual_python_replay_samples_to_select, len(replay_candidate_pool))
            selected_replay_samples = random.sample(replay_candidate_pool, num_to_select_replay)
            logger.info(f"Selected {len(selected_replay_samples)} Python samples for replay.")
            current_phase_selected_ids.update({s.get("snippet_id") for s in selected_replay_samples if s.get("snippet_id")})

    final_selected_samples = selected_main_samples + selected_replay_samples
    if final_selected_samples: 
        random.shuffle(final_selected_samples) 

    logger.info(f"Total samples selected for this phase: {len(final_selected_samples)}")
    if final_selected_samples:
        lang_counts = {}
        for s in final_selected_samples:
            lang = s.get("language", "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        logger.info(f"Language distribution in selected data for phase: {lang_counts}")
        
    return final_selected_samples, current_phase_selected_ids


# --- Create DataLoaders ---
def create_dataloaders(records, tokenizer, max_seq_len, batch_size, val_split_ratio): 
    logger.debug(f"create_dataloaders called with {len(records)} records.")
    if not records:
        logger.warning("Cannot create dataloaders from empty record list.")
        return None, None

    try:
        dataset = FineTuningDataset(records, tokenizer, max_seq_len)
    except Exception as e:
        logger.error(f"Error creating FineTuningDataset: {e}", exc_info=True)
        return None, None 
    
    if len(dataset) == 0: 
        logger.warning("FineTuningDataset is empty. Cannot create dataloaders.")
        return None, None

    train_dataset, val_dataset = dataset, None 

    if val_split_ratio > 0 and len(dataset) > 1 :
        val_size = int(len(dataset) * val_split_ratio)
        if val_size == 0 and len(dataset) > 1: val_size = 1
        
        train_size = len(dataset) - val_size
        
        if train_size <= 0 or val_size <=0 : 
            logger.warning(f"Train size ({train_size}) or Val size ({val_size}) is zero after split. Using all data for training.")
            train_dataset = dataset 
            val_dataset = None
        else:
            try:
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
            except Exception as e:
                logger.error(f"Error during random_split: {e}. Using all data for training.", exc_info=True)
                train_dataset = dataset
                val_dataset = None
    else:
        logger.info("Validation split ratio is 0 or dataset too small for split, using all data for training.")

    num_dataloader_workers = 2 if device.type == 'cuda' else 0 
    pin_memory_setting = True if device.type == 'cuda' else False

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=num_dataloader_workers, pin_memory=pin_memory_setting)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, 
                                num_workers=num_dataloader_workers, pin_memory=pin_memory_setting) if val_dataset and len(val_dataset) > 0 else None
    
    logger.info(f"Created Train Dataloader with {len(train_dataloader) if train_dataloader else 0} batches.")
    if val_dataloader:
        logger.info(f"Created Validation Dataloader with {len(val_dataloader)} batches.")
    else:
        logger.info("No Validation Dataloader created for this phase.")
        
    return train_dataloader, val_dataloader

# --- Training and Validation Function (GPU/CPU Adapted) ---
def fine_tune_phase(model, tokenizer, train_dataloader, val_dataloader, epochs, learning_rate, 
                    warmup_ratio, model_save_dir, phase_name, patience, 
                    freeze_config=None): 
    
    logger.info(f"Entering fine_tune_phase for '{phase_name}'. Train Dataloader: {len(train_dataloader) if train_dataloader else 'None'} batches.")

    if os.path.exists(model_save_dir):
        logger.info(f"Clearing previous best model directory for phase '{phase_name}': {model_save_dir}")
        try:
            shutil.rmtree(model_save_dir)
        except Exception as e:
            logger.error(f"Could not remove old model directory {model_save_dir}: {e}")
    try:
        os.makedirs(model_save_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create model save directory {model_save_dir}: {e}. Cannot save model for this phase.")
        return None

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.warning(f"No trainable parameters found for phase '{phase_name}'. Skipping training.")
        return None 

    optimizer = AdamW(trainable_params, lr=learning_rate)
    
    if not train_dataloader or len(train_dataloader) == 0:
        logger.warning(f"Train dataloader for phase '{phase_name}' is empty. Skipping fine-tuning.")
        return None 

    num_training_steps_per_epoch = len(train_dataloader)
    num_total_training_steps = num_training_steps_per_epoch * epochs
    if num_total_training_steps == 0: 
        logger.warning(f"Total training steps is 0 for phase '{phase_name}'. Skipping.")
        return None
        
    num_warmup_steps = int(warmup_ratio * num_total_training_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_total_training_steps
    )
    
    scaler = None
    use_amp = False 
    if device.type == 'cuda' and hasattr(torch.cuda.amp, 'GradScaler'):
        scaler = torch.cuda.amp.GradScaler()
        use_amp = True
        logger.info("CUDA device detected. Using Automatic Mixed Precision (AMP).")


    best_val_loss = float('inf')
    best_val_accuracy = 0.0 
    epochs_no_improve = 0
    best_model_path_this_phase = None

    logger.info(f"Starting fine-tuning phase: '{phase_name}' for {epochs} epoch(s). LR: {learning_rate}. Total steps: {num_total_training_steps}. Warmup steps: {num_warmup_steps}.")

    for epoch in range(epochs):
        logger.info(f"  Phase '{phase_name}' - Starting Epoch {epoch + 1}/{epochs}")
        
        if freeze_config: 
            should_reinit_optimizer = False
            if epoch < freeze_config["unfreeze_after_epoch"]:
                if not getattr(fine_tune_phase, f"_frozen_epoch_{epoch}_{phase_name}", False): 
                    model = freeze_model_layers(model, 
                                                freeze_config.get("num_embedding_layers_to_freeze",0),
                                                freeze_config.get("num_transformer_blocks_to_freeze",0))
                    logger.info(f"  Epoch {epoch+1}: Layers kept/set frozen as per freeze_config.")
                    setattr(fine_tune_phase, f"_frozen_epoch_{epoch}_{phase_name}", True)
                    should_reinit_optimizer = True 
            elif epoch == freeze_config["unfreeze_after_epoch"] and not getattr(fine_tune_phase, f"_unfrozen_for_phase_{phase_name}", False):
                logger.info(f"  Epoch {epoch+1}: Unfreezing layers as per freeze_config for phase '{phase_name}'.")
                model = unfreeze_all_layers(model)
                setattr(fine_tune_phase, f"_unfrozen_for_phase_{phase_name}", True)
                should_reinit_optimizer = True 
            
            if should_reinit_optimizer:
                logger.info("  Re-initializing optimizer and scheduler due to change in trainable layers.")
                trainable_params_now = [p for p in model.parameters() if p.requires_grad]
                if not trainable_params_now:
                    logger.error("  CRITICAL: No trainable parameters after layer freeze/unfreeze. Stopping phase.")
                    return None
                optimizer = AdamW(trainable_params_now, lr=learning_rate) 
                scheduler = get_linear_schedule_with_warmup( 
                    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_total_training_steps
                )
                for _ in range(epoch * num_training_steps_per_epoch): 
                    if scheduler.last_epoch < num_total_training_steps -1 : 
                         scheduler.step()

        model.train() 
        total_train_loss = 0
        num_train_batches = 0
        
        epoch_train_loader = train_dataloader

        train_progress_bar = tqdm(epoch_train_loader, desc=f"  Epoch {epoch + 1} Training", leave=False)

        for batch_idx, batch in enumerate(train_progress_bar):
            try:
                optimizer.zero_grad(set_to_none=True) # set_to_none=True can be slightly more efficient
                input_ids = batch["input_ids"].to(device) 
                attention_mask = batch["attention_mask"].to(device) 
                labels = input_ids.clone()
                if tokenizer.pad_token_id is not None:
                    labels[labels == tokenizer.pad_token_id] = -100 

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                    if loss is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else: 
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    if loss is not None:
                        loss.backward()
                        optimizer.step()
                
                if loss is not None:
                    scheduler.step() 
                    total_train_loss += loss.item()
                    num_train_batches += 1
                    train_progress_bar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
                else:
                    logger.warning(f"  Training (Batch {batch_idx}): Loss is None. Skipping backprop.")
            except RuntimeError as e_runtime: # Catch specific runtime errors like OOM
                if "out of memory" in str(e_runtime).lower():
                    logger.error(f"CUDA out of memory during training batch {batch_idx} in phase '{phase_name}', epoch {epoch+1}. Attempting to clear cache.", exc_info=False) # Less verbose OOM
                    if device.type == 'cuda': torch.cuda.empty_cache()
                    # Optionally, try to skip this batch or reduce batch size dynamically (more complex)
                    # For now, we just log and continue, which might lead to inaccurate epoch loss if many batches fail.
                    # Consider raising the error to stop if OOM is persistent.
                else:
                    logger.error(f"  Runtime error during training batch {batch_idx} in phase '{phase_name}', epoch {epoch+1}: {e_runtime}", exc_info=True)
                continue # Continue to next batch
            except Exception as e_batch:
                logger.error(f"  Error during training batch {batch_idx} in phase '{phase_name}', epoch {epoch+1}: {e_batch}", exc_info=True)
                continue 
        
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float('nan')
        logger.info(f"  Phase '{phase_name}' - Epoch {epoch + 1} Training finished. Average Loss: {avg_train_loss:.4f}")

        if val_dataloader and len(val_dataloader) > 0:
            logger.info(f"  Phase '{phase_name}' - Epoch {epoch + 1} Starting Validation...")
            model.eval() 
            total_val_loss = 0
            total_val_correct_preds = 0
            total_val_target_tokens = 0
            num_val_batches = 0
            
            epoch_val_loader = val_dataloader

            val_progress_bar = tqdm(epoch_val_loader, desc=f"  Epoch {epoch + 1} Validation", leave=False)
            
            with torch.no_grad():
                for batch_idx_val, batch_val in enumerate(val_progress_bar): 
                    try:
                        input_ids = batch_val["input_ids"].to(device) 
                        attention_mask = batch_val["attention_mask"].to(device) 
                        labels = input_ids.clone()
                        if tokenizer.pad_token_id is not None:
                            labels_for_loss = labels.clone()
                            labels_for_loss[labels_for_loss == tokenizer.pad_token_id] = -100 
                        else:
                            labels_for_loss = labels
                        
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_for_loss)
                        else:
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_for_loss)
                        
                        loss = outputs.loss
                        logits = outputs.logits

                        if loss is not None:
                            total_val_loss += loss.item()
                        else:
                            logger.warning(f"  Validation (Batch {batch_idx_val}): Loss is None.")
                        
                        active_loss_mask = attention_mask.view(-1) == 1 
                        active_logits = logits.view(-1, model.config.vocab_size)[active_loss_mask]
                        active_labels = labels.view(-1)[active_loss_mask] 

                        if active_logits.size(0) > 0: 
                            preds = torch.argmax(active_logits, dim=1)
                            total_val_correct_preds += (preds == active_labels).sum().item()
                            total_val_target_tokens += active_labels.size(0)
                        
                        num_val_batches +=1
                        val_progress_bar.set_postfix({'val_loss': loss.item() if loss else float('nan')})
                    except RuntimeError as e_runtime_val:
                        if "out of memory" in str(e_runtime_val).lower():
                            logger.error(f"CUDA out of memory during validation batch {batch_idx_val} in phase '{phase_name}', epoch {epoch+1}. Attempting to clear cache.", exc_info=False)
                            if device.type == 'cuda': torch.cuda.empty_cache()
                        else:
                            logger.error(f"  Runtime error during validation batch {batch_idx_val} in phase '{phase_name}', epoch {epoch+1}: {e_runtime_val}", exc_info=True)
                        continue
                    except Exception as e_val_batch:
                        logger.error(f"  Error during validation batch {batch_idx_val} in phase '{phase_name}', epoch {epoch+1}: {e_val_batch}", exc_info=True)
                        continue 
            
            avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
            avg_val_accuracy = total_val_correct_preds / total_val_target_tokens if total_val_target_tokens > 0 else 0.0
            logger.info(f"  Phase '{phase_name}' - Epoch {epoch + 1} Validation finished. Avg Loss: {avg_val_loss:.4f}, Avg Accuracy: {avg_val_accuracy:.4f}")

            if avg_val_loss < best_val_loss: 
                logger.info(f"  Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f} (Acc: {avg_val_accuracy:.4f}). Saving model to {model_save_dir}")
                best_val_loss = avg_val_loss
                best_val_accuracy = avg_val_accuracy 
                try:
                    model.save_pretrained(model_save_dir)
                    tokenizer.save_pretrained(model_save_dir)
                    best_model_path_this_phase = model_save_dir
                except Exception as e_save:
                    logger.error(f"  Error saving model during phase '{phase_name}': {e_save}", exc_info=True)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                logger.info(f"  Validation loss did not improve. Best: {best_val_loss:.4f} (Acc: {best_val_accuracy:.4f}). Current: {avg_val_loss:.4f} (Acc: {avg_val_accuracy:.4f}). Epochs no improve: {epochs_no_improve}/{patience}")
                if epochs_no_improve >= patience:
                    logger.info(f"  Early stopping triggered for phase '{phase_name}' after {epoch + 1} epochs.")
                    break 
        else: 
            logger.info(f"  Phase '{phase_name}' - Epoch {epoch + 1} No validation set. Saving model directly to {model_save_dir}.")
            try:
                model.save_pretrained(model_save_dir) 
                tokenizer.save_pretrained(model_save_dir)
                best_model_path_this_phase = model_save_dir
            except Exception as e_save:
                 logger.error(f"  Error saving model (no validation) during phase '{phase_name}': {e_save}", exc_info=True)
        
        if device.type == 'cuda': # Clear cache at end of epoch if on GPU
            torch.cuda.empty_cache()
            logger.debug(f"Cleared CUDA cache at end of epoch {epoch+1} for phase '{phase_name}'.")


    logger.info(f"Fine-tuning phase '{phase_name}' complete. Best validation loss for phase: {best_val_loss:.4f}, Accuracy: {best_val_accuracy:.4f}")
    
    if not best_model_path_this_phase and val_dataloader and epochs > 0 and num_train_batches > 0 : 
         logger.warning(f"No best model was saved during phase '{phase_name}' based on validation loss. The model in memory is the last state from epoch {epoch+1}.")
         last_state_path = os.path.join(model_save_dir, "last_state_after_no_improvement")
         try:
            os.makedirs(last_state_path, exist_ok=True)
            logger.info(f"Saving last model state of phase '{phase_name}' to {last_state_path}")
            model.save_pretrained(last_state_path)
            tokenizer.save_pretrained(last_state_path)
            return last_state_path 
         except Exception as e_save_last:
            logger.error(f"Error saving last model state for phase '{phase_name}': {e_save_last}", exc_info=True)
            return None 
    elif not val_dataloader and epochs > 0 and num_train_batches > 0: 
        best_model_path_this_phase = model_save_dir 

    return best_model_path_this_phase


def main():
    try:
        logger.info("🚀 Starting Stage 3: Advanced Phased and Validated Active Learning Cycle (GPU/CPU)...")
        print("--- PRINT: Stage 3 main() started. ---") 

        logger.info(f"Loading base tokenizer: {BASE_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Tokenizer pad_token was None. Set to eos_token: {tokenizer.eos_token}")
            else: 
                tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
                logger.warning("Tokenizer pad_token and eos_token were None. Added '[PAD]' as pad token.")
        
        current_model_path_or_name = BASE_MODEL_NAME 

        data_pool = load_data_pool(STAGE2_OUTPUT_POOL)
        if not data_pool:
            logger.error("Failed to load data pool or pool is empty. Exiting script.")
            return
        
        all_used_sample_ids_in_cycle = set() 

        for phase_idx, phase_config in enumerate(TRAINING_PHASES):
            phase_name = phase_config["phase_name"]
            phase_model_save_dir = os.path.join(OUTPUT_DIR_BASE, f"{phase_name}_best_model")

            logger.info("\n" + "="*30 + f" Starting Phase {phase_idx+1}: {phase_name} " + "="*30)
            print(f"\n--- PRINT: Starting Phase {phase_idx+1}: {phase_name} ---")
            logger.info(f"Configuration for phase '{phase_name}': {phase_config}")

            logger.info(f"Loading model from: {current_model_path_or_name} for phase '{phase_name}'")
            try:
                model = AutoModelForCausalLM.from_pretrained(current_model_path_or_name).to(device)
                if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
                    logger.info(f"Setting model.config.pad_token_id to tokenizer's pad_token_id: {tokenizer.pad_token_id}")
                    model.config.pad_token_id = tokenizer.pad_token_id
                if len(tokenizer) > model.config.vocab_size: 
                    logger.info(f"Resizing model token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
                    model.resize_token_embeddings(len(tokenizer))

            except Exception as e:
                logger.error(f"Failed to load model from {current_model_path_or_name}: {e}. Skipping phase.", exc_info=True)
                if device.type == 'cuda': torch.cuda.empty_cache() # Try to free memory
                continue
            
            selected_data, newly_selected_ids = select_data_for_phase(
                data_pool, 
                phase_config["num_samples"], 
                language_filter=phase_config.get("language_filter"), 
                python_replay_ratio=phase_config.get("python_replay_ratio", 0.0),
                previously_selected_ids=all_used_sample_ids_in_cycle
            )
            
            if not selected_data:
                logger.warning(f"No data selected for phase '{phase_name}'. Skipping phase.")
                if device.type == 'cuda': torch.cuda.empty_cache()
                continue 

            all_used_sample_ids_in_cycle.update(newly_selected_ids)

            train_loader, val_loader = create_dataloaders(
                selected_data, tokenizer, MAX_SEQ_LENGTH, BATCH_SIZE, VALIDATION_SPLIT_RATIO
            )

            if not train_loader or len(train_loader) == 0 : 
                logger.warning(f"Train loader for phase '{phase_name}' is empty. Skipping fine-tuning for this phase.")
                if device.type == 'cuda': torch.cuda.empty_cache()
                continue

            initial_freeze_config = phase_config.get("freeze_layers_config")
            if initial_freeze_config:
                if initial_freeze_config.get("num_transformer_blocks_to_freeze", 0) > 0 or \
                   initial_freeze_config.get("num_embedding_layers_to_freeze", 0) > 0:
                    logger.info(f"Applying initial layer freezing for phase '{phase_name}'.")
                    model = freeze_model_layers(model, 
                                                initial_freeze_config.get("num_embedding_layers_to_freeze",0),
                                                initial_freeze_config.get("num_transformer_blocks_to_freeze",0))
            else: 
                logger.info(f"Ensuring all layers are trainable for phase '{phase_name}' (no initial freeze_config).")
                model = unfreeze_all_layers(model) 

            best_model_path_from_phase = fine_tune_phase(
                model, tokenizer, train_loader, val_loader,
                epochs=phase_config["epochs"], 
                learning_rate=phase_config["learning_rate"],
                warmup_ratio=phase_config["warmup_ratio"], 
                model_save_dir=phase_model_save_dir,
                phase_name=phase_name, 
                patience=EARLY_STOPPING_PATIENCE,
                freeze_config=initial_freeze_config 
            )

            if best_model_path_from_phase:
                current_model_path_or_name = best_model_path_from_phase 
                logger.info(f"Phase '{phase_name}' complete. Best model for this phase at: {current_model_path_or_name}")
            else:
                logger.warning(f"Phase '{phase_name}' did not result in a saved best model. The model from '{current_model_path_or_name}' will be used for the next phase if applicable.")
            
            # Clear CUDA cache after a phase completes or fails to save a model
            if device.type == 'cuda':
                logger.info(f"Clearing CUDA cache after phase '{phase_name}'.")
                del model # Explicitly delete model from previous phase from memory
                if 'train_loader' in locals() and train_loader is not None: del train_loader
                if 'val_loader' in locals() and val_loader is not None: del val_loader
                torch.cuda.empty_cache()


        logger.info(f"✅ All training phases complete for this cycle. Final model considered is at: {current_model_path_or_name}")

    except Exception as e_main:
        logger.error("CRITICAL ERROR in main execution:", exc_info=True)
        print(f"CRITICAL ERROR in main execution: {e_main}") 
    finally:
        logging.shutdown() 
        print("--- PRINT: Stage 3 main() finished or errored out. ---")


if __name__ == "__main__":
    print("--- PRINT: Stage 3 script __main__ block reached. ---")
    main()

