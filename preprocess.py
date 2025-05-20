# preprocess.py (for Active Learning Pipeline - Stage 1)

import os
import json
from datasets import load_dataset, IterableDataset
from tqdm.auto import tqdm
import logging
import ast # For a very basic Python code viability check

# --- Configuration ---
# Output directory for this stage's processed JSONL files
PROCESSED_JSONL_DIR = "./processed_jsonl_stage1"
os.makedirs(PROCESSED_JSONL_DIR, exist_ok=True)

# Minimum length for raw_content to be considered viable
MIN_CONTENT_LENGTH = 20 # Characters

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Datasets Configuration ---
# Define datasets to process.
# 'id': A unique identifier for this dataset configuration.
# 'hf_name': Hugging Face dataset name.
# 'subset': Specific subset/data_dir if applicable (e.g., for 'the-stack-dedup').
# 'language': The programming language for this data.
# 'split': Dataset split to use (e.g., 'train').
# 'content_fields': List of field names to try for extracting the main code/text content.
#                   The script will use the first field found that contains non-empty string data.
# 'combine_fields_separator': If multiple fields are specified and found, how to join them.
#                             Set to None if only one field should be used or if specific handling is needed.
# 'solution_language_filter': For datasets like APPS where solutions are multi-lingual within a record.
#                             Set to the language to filter for (e.g., "python").
# 'max_samples': Maximum number of samples to process from this dataset.
# 'stream': Whether to use streaming (load_dataset(..., stream=True)). Recommended for large datasets.
# 'trust_remote_code': Set to True if the dataset loading requires it.
DATASETS_CONFIG = [
    # --- Python Foundation ---
    {"id": "mbpp_py", "hf_name": "mbpp", "subset": None, "language": "python", "split": "train",
     "content_fields": ["text", "code"], "combine_fields_separator": "\n\n# Solution:\n",
     "max_samples": 10000, "stream": False, "trust_remote_code": False},
    {"id": "humaneval_py", "hf_name": "openai_humaneval", "subset": None, "language": "python", "split": "test",
     "content_fields": ["prompt", "canonical_solution"], "combine_fields_separator": "\n",
     "max_samples": 2000, "stream": False, "trust_remote_code": False}, # Test split has problems
    {"id": "apps_py", "hf_name": "codeparrot/apps", "subset": None, "language": "python", "split": "train",
     "content_fields": ["problem", "solutions"], "combine_fields_separator": "\n\n# Solutions:\n",
     "solution_language_filter": "python", # Special handling for 'solutions' field
     "max_samples": 20000, "stream": False, "trust_remote_code": False}, # Reduced from 50k for initial run
    {"id": "codeforces_py", "hf_name": "MatrixStudio/Codeforces-Python-Submissions", "subset": None, "language": "python", "split": "train",
     "content_fields": ["code"], "combine_fields_separator": None,
     "max_samples": 50000, "stream": False, "trust_remote_code": False}, # Reduced from 100k
    {"id": "alpaca_py_instr", "hf_name": "iamtarun/python_code_instructions_18k_alpaca", "subset": None, "language": "python", "split": "train",
     "content_fields": ["instruction", "output"], "combine_fields_separator": "\n\n# Output:\n",
     "max_samples": 18000, "stream": False, "trust_remote_code": False}, # Max is ~18k
    {"id": "the_stack_py_small", "hf_name": "bigcode/the-stack-dedup", "subset": "data/python", "language": "python", "split": "train",
     "content_fields": ["content"], "combine_fields_separator": None,
     "max_samples": 200000, "stream": True, "trust_remote_code": False}, # Reduced from 500k

    # --- C++ ---
    {"id": "the_stack_cpp_small", "hf_name": "bigcode/the-stack-dedup", "subset": "data/cpp", "language": "cpp", "split": "train",
     "content_fields": ["content"], "combine_fields_separator": None,
     "max_samples": 100000, "stream": True, "trust_remote_code": False}, # Reduced from 200k

    # --- Java ---
    {"id": "the_stack_java_small", "hf_name": "bigcode/the-stack-dedup", "subset": "data/java", "language": "java", "split": "train",
     "content_fields": ["content"], "combine_fields_separator": None,
     "max_samples": 100000, "stream": True, "trust_remote_code": False}, # Reduced from 200k

    # --- Rust ---
    {"id": "the_stack_rust_small", "hf_name": "bigcode/the-stack-dedup", "subset": "data/rust", "language": "rust", "split": "train",
     "content_fields": ["content"], "combine_fields_separator": None,
     "max_samples": 50000, "stream": True, "trust_remote_code": False}, # Reduced from 100k

    # --- C ---
    {"id": "the_stack_c_small", "hf_name": "bigcode/the-stack-dedup", "subset": "data/c", "language": "c", "split": "train",
     "content_fields": ["content"], "combine_fields_separator": None,
     "max_samples": 50000, "stream": True, "trust_remote_code": False}, # Reduced from 100k
]

def get_content_from_example(example, config):
    """
    Extracts and combines content from specified fields in an example.
    Handles special case for APPS 'solutions' field.
    """
    contents_to_join = []
    
    if config["id"].startswith("apps_") and "solutions" in config["content_fields"]:
        # Handle APPS 'solutions' field which is a list of strings (JSON encoded)
        # and needs to be filtered by language if solution_language_filter is set.
        # We'll take the problem description and append Python solutions.
        problem_desc = ""
        if "problem" in example and isinstance(example["problem"], str) and example["problem"].strip():
            problem_desc = example["problem"].strip()
        
        solutions_text = ""
        if "solutions" in example and isinstance(example["solutions"], list):
            # Assuming solutions are JSON strings that need to be parsed if they are not already dicts/lists of strings
            # For APPS, 'solutions' are typically lists of strings.
            # We are looking for Python solutions.
            # This part might need adjustment based on the exact structure of APPS 'solutions'.
            # For now, let's assume it's a list of code strings.
            # A more robust way would be to check if the dataset loader already parsed them.
            # If `solution_language_filter` is Python, we'd ideally filter here.
            # For simplicity, we'll join them for now. Stage 2 can do more advanced filtering.
            # This example assumes solutions are directly usable strings.
            if config.get("solution_language_filter") == "python":
                 # This is a placeholder; APPS dataset solutions are not tagged by language *within* the list.
                 # The dataset itself is often filtered by language when loading or has language-specific versions.
                 # If solutions were dicts with a 'language' key, we could filter.
                 # For now, we assume if config["language"] is python, all solutions are python.
                python_solutions = [s for s in example["solutions"] if isinstance(s, str)] # Basic check
                if python_solutions:
                    solutions_text = f"\n# Python Solutions:\n" + "\n\n---\n\n".join(python_solutions)

        if problem_desc:
            return problem_desc + solutions_text if solutions_text else problem_desc
        return None # Or just solutions_text if no problem_desc

    # General case for other datasets
    for field in config["content_fields"]:
        if field in example and isinstance(example[field], str) and example[field].strip():
            contents_to_join.append(example[field].strip())
        elif field in example and isinstance(example[field], list) and all(isinstance(i, str) for i in example[field]):
            # If field is a list of strings, join them
            contents_to_join.append("\n".join(item.strip() for item in example[field] if item.strip()))


    if not contents_to_join:
        return None
    
    if config["combine_fields_separator"] is not None and len(contents_to_join) > 1:
        return config["combine_fields_separator"].join(contents_to_join)
    elif contents_to_join:
        return contents_to_join[0] # Use the first one found if no separator or only one field
    return None


def is_viable_python_code(code_content):
    """
    A very basic check to see if the content might be Python code
    by trying to parse it with ast.
    Returns True if parsable or not Python, False if it's Python and fails to parse.
    """
    try:
        ast.parse(code_content)
        return True  # Parsable, so it's valid Python syntax
    except (SyntaxError, ValueError, TypeError, MemoryError): # Added more exceptions
        return False # Fails to parse, likely not valid standalone Python

def main():
    logger.info("🚀 Starting Stage 1: Raw Data Ingestion and Initial Cleaning...")
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
            # Load dataset
            ds_args = {"path": config["hf_name"], "split": config["split"], "streaming": config["stream"]}
            if config["subset"]:
                ds_args["data_dir"] = config["subset"]
            if config.get("trust_remote_code"):
                ds_args["trust_remote_code"] = config["trust_remote_code"]
            
            dataset = load_dataset(**ds_args)

            if config["stream"] and isinstance(dataset, IterableDataset):
                dataset_iterable = dataset.take(config["max_samples"])
                # For streamed datasets, getting total count upfront is not straightforward.
                # We'll count as we go.
            else: # Not streaming or not an IterableDataset (though load_dataset with stream=True should yield IterableDataset)
                # If not streaming, we can shuffle and select if desired, but for now, just take head.
                if len(dataset) > config["max_samples"]:
                     # This might be slow for very large non-streamed datasets.
                     # Consider dataset.select(range(config["max_samples"]))
                    dataset_iterable = dataset.select(range(config["max_samples"]))
                else:
                    dataset_iterable = dataset
                logger.info(f"  Total raw examples available (or selected): {len(dataset_iterable)} for {config['id']}")


            with open(output_file_path, "w", encoding="utf-8") as outfile:
                for example in tqdm(dataset_iterable, desc=f"  Iterating {config['id']}", unit=" examples"):
                    processed_count += 1
                    raw_content = get_content_from_example(example, config)

                    if not raw_content or len(raw_content) < MIN_CONTENT_LENGTH:
                        continue

                    # Basic viability check specifically for Python code using AST parsing
                    # For other languages, this check is skipped.
                    if config["language"] == "python":
                        if not is_viable_python_code(raw_content):
                            # This might be too aggressive if 'raw_content' includes prompts + code.
                            # Consider applying only if content is expected to be *just* code.
                            # For now, we'll log it if it fails.
                            # logger.debug(f"    Skipping non-viable Python AST for content: {raw_content[:100]}...")
                            pass # Keep it for now, Stage 2 can do more advanced checks

                    # Try to get an original ID if available, otherwise use processed_count
                    original_id = example.get("id", example.get("example_id", f"{config['id']}_{processed_count}"))


                    record = {
                        "original_id": str(original_id), # Ensure ID is string
                        "source_config_id": config["id"], # Which config entry this came from
                        "source_hf_name": config["hf_name"],
                        "language": config["language"],
                        "raw_content": raw_content
                    }
                    outfile.write(json.dumps(record) + "\n")
                    written_count += 1

                    if written_count >= config["max_samples"] and not config["stream"]: # Already handled by .take for stream
                        break
            
            logger.info(f"  Finished {config['id']}. Processed: {processed_count}. Written: {written_count} to {output_file_path}")

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
