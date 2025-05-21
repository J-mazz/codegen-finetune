# preprocess.py (for Active Learning Pipeline - Stage 1 - New C & Rust Datasets)

import os
import json
from datasets import load_dataset, IterableDataset # IterableDataset might not be used if all stream=False
from tqdm.auto import tqdm
import logging
import ast # For a very basic Python code viability check

# --- Configuration ---
PROCESSED_JSONL_DIR = "./processed_jsonl_stage1"
os.makedirs(PROCESSED_JSONL_DIR, exist_ok=True)
MIN_CONTENT_LENGTH = 20 # Characters

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Datasets Configuration (New C & Rust Datasets) ---
DATASETS_CONFIG = [
    # --- Python Foundation (All Non-Streaming) ---
    {"id": "mbpp_py", "hf_name": "mbpp", "subset": None, "language": "python", "split": "train",
     "content_fields": ["text", "code"], "combine_fields_separator": "\n\n# Solution:\n",
     "max_samples": 10000, "stream": False, "trust_remote_code": False},
    {"id": "humaneval_py", "hf_name": "openai_humaneval", "subset": None, "language": "python", "split": "test",
     "content_fields": ["prompt", "canonical_solution"], "combine_fields_separator": "\n",
     "max_samples": 2000, "stream": False, "trust_remote_code": False},
    {"id": "apps_py", "hf_name": "codeparrot/apps", "subset": None, "language": "python", "split": "train",
     "content_fields": ["problem", "solutions"], "combine_fields_separator": "\n\n# Solutions:\n",
     "solution_language_filter": "python", # Special handling in get_content_from_example
     "max_samples": 50, "stream": False, "trust_remote_code": False},
    {"id": "codeforces_py", "hf_name": "MatrixStudio/Codeforces-Python-Submissions", "subset": None, "language": "python", "split": "train",
     "content_fields": ["code"], "combine_fields_separator": None,
     "max_samples": 50000, "stream": False, "trust_remote_code": False},
    {"id": "alpaca_py_instr", "hf_name": "iamtarun/python_code_instructions_18k_alpaca", "subset": None, "language": "python", "split": "train",
     "content_fields": ["instruction", "output"], "combine_fields_separator": "\n\n# Output:\n",
     "max_samples": 18000, "stream": False, "trust_remote_code": False},

    # --- C++ (Using deepmind/code_contests) ---
    {"id": "cpp_code_contests",
     "hf_name": "deepmind/code_contests",
     "subset": None, 
     "language": "cpp", "split": "train", 
     "content_fields": ["description"], 
     "solutions_field": "solutions", 
     "solution_lang_key": "language", 
     "solution_code_key": "solution", 
     "combine_fields_separator": "\n\n// Solution:\n",
     "max_samples": 10000, "stream": False, "trust_remote_code": True},

    # --- Java (Using deepmind/code_contests) ---
    {"id": "java_code_contests",
     "hf_name": "deepmind/code_contests",
     "subset": None,
     "language": "java", "split": "train",
     "content_fields": ["description"],
     "solutions_field": "solutions", "solution_lang_key": "language", "solution_code_key": "solution",
     "combine_fields_separator": "\n\n// Solution:\n",
     "max_samples": 10000, "stream": False, "trust_remote_code": True},
    
    # --- Rust (New Suggested Dataset) ---
    {"id": "rust_code_snippets_new",
     "hf_name": "RustHint/Rust_Code_Snippets", # New suggestion - VERIFY this dataset and its fields
     "subset": None,
     "language": "rust", "split": "train", # VERIFY split name
     "content_fields": ["Snippet"], # VERIFY THIS FIELD NAME for this dataset (e.g., "Snippet", "code", "text")
     "combine_fields_separator": None,
     "max_samples": 5000, "stream": False, "trust_remote_code": False},

    # --- C (New Suggested Dataset) ---
    {"id": "c_code_snippets_new",
     "hf_name": "nampdn-ai/C-Code-Snippets", # New suggestion - VERIFY this dataset and its fields
     "subset": None,
     "language": "c", "split": "train", # VERIFY split name
     "content_fields": ["Text"], # VERIFY THIS FIELD NAME for this dataset (e.g., "Text", "code")
     "combine_fields_separator": None,
     "max_samples": 10000, "stream": False, "trust_remote_code": False},
]

def get_content_from_example(example, config):
    contents_to_join = []
    description_part = ""
    solution_part = ""

    # Handle problem description using content_fields (usually the first one)
    if config.get("content_fields"):
        for field in config["content_fields"]:
            if field in example and isinstance(example[field], str) and example[field].strip():
                description_part = example[field].strip()
                break # Take the first valid description field

    # Special handling for datasets with structured solutions (like deepmind/code_contests)
    if config["hf_name"] == "deepmind/code_contests":
        solutions_field_name = config.get("solutions_field", "solutions")
        lang_key = config.get("solution_lang_key", "language")
        code_key = config.get("solution_code_key", "solution")
        target_language_lower = config["language"].lower()
        
        if solutions_field_name in example and isinstance(example[solutions_field_name], list):
            for sol_dict in example[solutions_field_name]:
                if isinstance(sol_dict, dict) and \
                   lang_key in sol_dict and \
                   code_key in sol_dict and \
                   isinstance(sol_dict[code_key], str) and \
                   sol_dict[code_key].strip():
                    
                    sol_lang_field = sol_dict[lang_key]
                    lang_match = False
                    if isinstance(sol_lang_field, str):
                        normalized_sol_lang = sol_lang_field.lower().replace("3", "").replace(" ", "")
                        if "c++" in normalized_sol_lang: normalized_sol_lang = "cpp"
                        if normalized_sol_lang == target_language_lower:
                            lang_match = True
                    elif isinstance(sol_lang_field, list): 
                        for lang_item in sol_lang_field:
                            if isinstance(lang_item, str):
                                normalized_sol_lang = lang_item.lower().replace("3", "").replace(" ", "")
                                if "c++" in normalized_sol_lang: normalized_sol_lang = "cpp"
                                if normalized_sol_lang == target_language_lower:
                                    lang_match = True
                                    break
                    if lang_match:
                        solution_part = sol_dict[code_key].strip()
                        break 
        if not solution_part:
            return None

    elif config["id"].startswith("apps_py"): 
        if "problem" in example and isinstance(example["problem"], str) and example["problem"].strip():
            description_part = example["problem"].strip()
        
        if "solutions" in example and isinstance(example["solutions"], list):
            parsed_python_solutions = []
            for sol_item in example["solutions"]:
                if isinstance(sol_item, str):
                    try:
                        actual_solutions_list = json.loads(sol_item)
                        if isinstance(actual_solutions_list, list):
                           parsed_python_solutions.extend([s for s in actual_solutions_list if isinstance(s, str)])
                    except json.JSONDecodeError: 
                        if not getattr(get_content_from_example, f"_logged_apps_json_error_{config['id']}", False):
                            logger.warning(f"APPS ({config['id']}): Could not decode one or more solution JSON strings. Example: {sol_item[:100]}. Further similar warnings will be suppressed.")
                            setattr(get_content_from_example, f"_logged_apps_json_error_{config['id']}", True)
                elif isinstance(sol_item, list): 
                    parsed_python_solutions.extend([s for s in sol_item if isinstance(s, str)])
            if parsed_python_solutions:
                solution_part = "\n\n---\n\n".join(parsed_python_solutions)
        if not description_part and not solution_part: return None

    else: # General case: extract from content_fields (This will apply to the new C and Rust datasets)
        temp_contents = []
        for field in config.get("content_fields", []): # Ensure content_fields is treated as a list
            if field in example and isinstance(example[field], str) and example[field].strip():
                temp_contents.append(example[field].strip())
            elif field in example and isinstance(example[field], list) and all(isinstance(i, str) for i in example[field]):
                temp_contents.append("\n".join(item.strip() for item in example[field] if item.strip()))
        
        if not temp_contents: return None # If no content found in specified fields
        
        # For simpler datasets, assume the first (or only) content_field is the main code/text
        # If description_part is already set (e.g. for code_contests), this won't overwrite it.
        # If description_part is empty, this becomes the main content.
        if not description_part and temp_contents: 
            description_part = temp_contents[0] 
            # If by chance multiple fields were listed and all had content, and it's not a structured solution type,
            # we might just take the first, or log a warning and combine.
            if len(temp_contents) > 1 : 
                logger.warning(f"Dataset {config['id']} has multiple content_fields and is not a known structured type (like code_contests/apps). Using content from '{config.get('content_fields')[0]}'. Other fields: {config.get('content_fields')[1:]}")
                # Or, if combining is desired for these general cases:
                # description_part = "\n".join(temp_contents)

    # Combine description and solution if both exist and a separator is defined
    if description_part and solution_part and config.get("combine_fields_separator"):
        return description_part + config["combine_fields_separator"] + solution_part
    elif solution_part: # If only solution part was populated (e.g. for code_contests if description was empty)
        return solution_part
    elif description_part: # If only description part (common for datasets with just one content field)
        return description_part
    
    return None


def is_viable_python_code(code_content):
    try:
        ast.parse(code_content)
        return True
    except (SyntaxError, ValueError, TypeError, MemoryError):
        return False

def main():
    logger.info("🚀 Starting Stage 1: Raw Data Ingestion and Initial Cleaning (New C & Rust Datasets)...")
    total_records_processed_all = 0
    total_records_written_all = 0

    for config in DATASETS_CONFIG:
        logger.info(f"➡️ Processing dataset: {config['id']} ({config['hf_name']}) for language: {config['language']}")
        
        lang_output_dir = os.path.join(PROCESSED_JSONL_DIR, config["language"])
        os.makedirs(lang_output_dir, exist_ok=True)
        output_file_path = os.path.join(lang_output_dir, f"{config['id']}.jsonl")

        processed_count = 0
        written_count = 0
        
        try:
            current_stream_setting = config.get("stream", False)
            if config['language'] != 'python' and current_stream_setting:
                logger.info(f"  Overriding stream to False for {config['id']} ({config['language']}) as per non-streaming preference.")
                current_stream_setting = False

            ds_args = {"path": config["hf_name"], "split": config["split"], "streaming": current_stream_setting}
            
            if config.get("subset"): 
                ds_args["name"] = config["subset"] # 'name' is often used for specific configurations within a dataset
            
            if config.get("trust_remote_code"):
                ds_args["trust_remote_code"] = config["trust_remote_code"]
            
            logger.info(f"  Attempting to load dataset with args: {ds_args}")
            dataset = load_dataset(**ds_args)
            logger.info(f"  Dataset loaded. Type: {type(dataset)}")

            first_example = None
            dataset_iterable_main_loop = None 

            try:
                if current_stream_setting and isinstance(dataset, IterableDataset):
                    peek_iter = dataset.take(1)
                    first_example = next(iter(peek_iter), None)
                    dataset_iterable_main_loop = dataset.take(config["max_samples"]) 
                elif not current_stream_setting: 
                    if len(dataset) > 0:
                        first_example = dataset[0]
                        num_to_select = min(config["max_samples"], len(dataset))
                        dataset_iterable_main_loop = dataset.select(range(num_to_select))
                    else: 
                        logger.warning(f"  Dataset {config['id']} is empty after loading. Skipping.")
                        continue
                else: 
                     logger.warning(f"  Unexpected dataset type or state for {config['id']}. Type: {type(dataset)}. Skipping.")
                     continue
            except StopIteration: 
                logger.warning(f"  Dataset {config['id']} (streamed) yielded no items for inspection (or is empty). Skipping.")
                continue
            except Exception as e_inspect:
                logger.warning(f"  Could not retrieve first example from {config['id']} for inspection: {e_inspect}. Skipping.")
                continue
            
            if not first_example:
                 logger.warning(f"  Dataset {config['id']} is effectively empty or first example could not be fetched. Skipping.")
                 continue

            primary_content_fields = config.get("content_fields", [])
            missing_fields = [f for f in primary_content_fields if f not in first_example]
            
            if config["hf_name"] == "deepmind/code_contests": # Check structure for code_contests
                solutions_field = config.get("solutions_field", "solutions")
                if solutions_field not in first_example:
                    missing_fields.append(solutions_field)
                elif isinstance(first_example.get(solutions_field), list) and len(first_example[solutions_field]) > 0:
                    sol_dict_example = first_example[solutions_field][0]
                    if not isinstance(sol_dict_example, dict):
                         missing_fields.append(f"{solutions_field} items are not dicts")
                    else:
                        if config.get("solution_lang_key") not in sol_dict_example:
                            missing_fields.append(config.get("solution_lang_key"))
                        if config.get("solution_code_key") not in sol_dict_example:
                            missing_fields.append(config.get("solution_code_key"))
            
            if missing_fields:
                logger.error(f"  Skipping dataset {config['id']} due to missing fields in the first example: {missing_fields}.")
                logger.error(f"    Available fields in first example: {list(first_example.keys())}")
                continue
            
            if dataset_iterable_main_loop is None: 
                logger.error(f"  Dataset iterable for {config['id']} was not properly initialized. Skipping.")
                continue

            logger.info(f"  Processing examples for {config['id']} (Max: {config['max_samples']})")

            with open(output_file_path, "w", encoding="utf-8") as outfile:
                iterator_for_tqdm = tqdm(dataset_iterable_main_loop, desc=f"  Iterating {config['id']}", unit=" examples", leave=False)
                
                for example in iterator_for_tqdm:
                    processed_count += 1
                    raw_content = get_content_from_example(example, config)

                    if not raw_content or len(raw_content) < MIN_CONTENT_LENGTH:
                        continue
                    
                    if config["language"] == "python":
                        if not is_viable_python_code(raw_content):
                            pass

                    original_id = example.get("id", example.get("name", example.get("problem_id", f"{config['id']}_{processed_count}")))
                    record = {
                        "original_id": str(original_id),
                        "source_config_id": config["id"],
                        "source_hf_name": config["hf_name"],
                        "language": config["language"],
                        "raw_content": raw_content
                    }
                    outfile.write(json.dumps(record) + "\n")
                    written_count += 1
            
            logger.info(f"  Finished {config['id']}. Processed: {processed_count}. Written: {written_count} to {output_file_path}")
            if written_count == 0 and processed_count > 0:
                 logger.warning(f"  Warning: Processed {processed_count} for {config['id']} but wrote 0 records. Check content extraction, filtering, or language matching (e.g., for code_contests).")

        except Exception as e:
            logger.error(f"❌ Error processing {config['id']}: {e}", exc_info=True)
        
        total_records_processed_all += processed_count
        total_records_written_all += written_count

    logger.info(f"\n🎉 All datasets processed for Stage 1.")
    logger.info(f"Total records iterated across all datasets: {total_records_processed_all}")
    logger.info(f"Total records written to JSONL files: {total_records_written_all}")
    logger.info(f"Output files are in: {PROCESSED_JSONL_DIR}")

if __name__ == "__main__":
    main()
