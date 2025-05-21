# stage2_characterize_dedup.py (for Active Learning Pipeline - Stage 2)

import os
import json
import hashlib
from tqdm.auto import tqdm
import logging
import ast # For Python AST analysis

# --- Configuration (Paths) ---
STAGE1_OUTPUT_DIR = "./processed_jsonl_stage1" # Input directory from Stage 1 (Reverted to original descriptive name)
FINAL_OUTPUT_JSONL = "./active_learn_pool.jsonl" # Output file for this stage (Concise)
LOG_FILE = "./s2_char_dedup.log" # Log file (Concise)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'), # Ensure log is overwritten for fresh run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Python Code Characterization ---
def characterize_python(code_content):
    """Characterizes Python code using AST."""
    characteristics = {
        "loc": len(code_content.splitlines()),
        "num_functions": 0,
        "num_classes": 0,
        "imports": [],
        "ast_parse_success": False,
        "syntax_error_detail": None,
        "avg_line_length": 0,
        "num_comments": 0 # Basic comment count
    }
    if characteristics["loc"] > 0:
        characteristics["avg_line_length"] = len(code_content) / characteristics["loc"]

    try:
        tree = ast.parse(code_content)
        characteristics["ast_parse_success"] = True
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                characteristics["num_functions"] += 1
            elif isinstance(node, ast.ClassDef):
                characteristics["num_classes"] += 1
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    characteristics["imports"].append(alias.name)
        characteristics["imports"] = list(set(characteristics["imports"])) # Unique imports
        
        # Basic comment count (can be improved with more sophisticated parsing)
        # This counts lines starting with #, might miss block comments or inline comments.
        characteristics["num_comments"] = sum(1 for line in code_content.splitlines() if line.strip().startswith("#"))

    except SyntaxError as e:
        characteristics["syntax_error_detail"] = f"Line {e.lineno}, Offset {e.offset}: {e.msg}"
        logger.debug(f"SyntaxError parsing Python content: Line {e.lineno}, {e.msg}. Content snippet: {code_content[:200]}")
    except Exception as e: # Catch other AST parsing errors
        characteristics["syntax_error_detail"] = str(e)
        logger.debug(f"Exception parsing Python content: {str(e)}. Content snippet: {code_content[:200]}")
        
    return characteristics

# --- C/C++ Code Characterization (Basic) ---
def characterize_c_cpp(code_content):
    """Basic characterization for C/C++ code."""
    lines = code_content.splitlines()
    loc = len(lines)
    characteristics = {
        "loc": loc,
        "includes": sum(1 for line in lines if line.strip().startswith("#include")),
        "main_function": any("main(" in line for line in lines), # Very basic check
        "function_defs": sum(1 for line in lines if ("(" in line and ")" in line and "{" in line and not line.strip().startswith("#") and not " class " in line and not " struct " in line)), # Heuristic
        "avg_line_length": len(code_content) / loc if loc > 0 else 0,
        "num_comments_sl": sum(1 for line in lines if line.strip().startswith("//")),
        "num_comments_ml_lines": sum(1 for line in lines if "/*" in line or "*/" in line) # counts lines involved in ml comments
    }
    return characteristics

# --- Java Code Characterization (Basic) ---
def characterize_java(code_content):
    """Basic characterization for Java code."""
    lines = code_content.splitlines()
    loc = len(lines)
    characteristics = {
        "loc": loc,
        "imports": sum(1 for line in lines if line.strip().startswith("import ")),
        "class_defs": sum(1 for line in lines if "class " in line and "{" in line),
        "main_method": any("public static void main(String[] args)" in line for line in lines), # Basic
        "avg_line_length": len(code_content) / loc if loc > 0 else 0,
        "num_comments_sl": sum(1 for line in lines if line.strip().startswith("//")),
        "num_comments_ml_lines": sum(1 for line in lines if "/*" in line or "*/" in line)
    }
    return characteristics

# --- Rust Code Characterization (Basic) ---
def characterize_rust(code_content):
    """Basic characterization for Rust code."""
    lines = code_content.splitlines()
    loc = len(lines)
    characteristics = {
        "loc": loc,
        "uses": sum(1 for line in lines if line.strip().startswith("use ")),
        "fn_defs": sum(1 for line in lines if line.strip().startswith("fn ") or " fn " in line),
        "struct_defs": sum(1 for line in lines if line.strip().startswith("struct ")),
        "impl_blocks": sum(1 for line in lines if line.strip().startswith("impl ") or " impl " in line),
        "main_function": any("fn main()" in line for line in lines), # Basic
        "avg_line_length": len(code_content) / loc if loc > 0 else 0,
        "num_comments_sl": sum(1 for line in lines if line.strip().startswith("//")),
    }
    return characteristics

def get_characterization_function(language):
    """Returns the appropriate characterization function based on language."""
    if language == "python":
        return characterize_python
    elif language == "cpp" or language == "c": # Using same basic func for C and C++
        return characterize_c_cpp
    elif language == "java":
        return characterize_java
    elif language == "rust":
        return characterize_rust
    else:
        # Generic fallback: just LOC and avg line length
        logger.debug(f"No specific characterization function for language '{language}'. Using generic.")
        return lambda code: {
            "loc": len(code.splitlines()),
            "avg_line_length": len(code) / len(code.splitlines()) if code.splitlines() else 0
        }

def main():
    # ---- DEBUG PRINT STATEMENT ----
    print(f"DEBUG: Stage 2 script - main() function entered. CWD: {os.getcwd()}")
    logger.info("🚀 Starting Stage 2: De-duplication, Characterization, and Unlabeled Pool Creation...")
    
    all_records = []
    # Iterate through language subdirectories created by Stage 1
    if not os.path.exists(STAGE1_OUTPUT_DIR):
        logger.error(f"Stage 1 output directory not found: {STAGE1_OUTPUT_DIR}")
        logger.error("Please run Stage 1 script (preprocess.py) first.")
        return

    for lang_dir in os.listdir(STAGE1_OUTPUT_DIR):
        lang_path = os.path.join(STAGE1_OUTPUT_DIR, lang_dir)
        if os.path.isdir(lang_path):
            logger.info(f"  Scanning language directory: {lang_dir}")
            for filename in os.listdir(lang_path):
                if filename.endswith(".jsonl"):
                    file_path = os.path.join(lang_path, filename)
                    logger.info(f"    Loading records from: {file_path}")
                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            for line_num, line in enumerate(infile):
                                try:
                                    record = json.loads(line)
                                    all_records.append(record)
                                except json.JSONDecodeError:
                                    logger.warning(f"      Skipping malformed JSON line {line_num+1} in {file_path}")
                    except Exception as e:
                        logger.error(f"    Error reading file {file_path}: {e}")
    
    logger.info(f"  Total records loaded from Stage 1: {len(all_records)}")
    if not all_records:
        logger.warning("No records loaded from Stage 1. Exiting.")
        return

    # De-duplication based on 'raw_content'
    logger.info("  De-duplicating records based on 'raw_content'...")
    unique_contents = {} # Using dict for de-duplication: hash -> record
    duplicates_found = 0
    for record in tqdm(all_records, desc="  De-duplicating", unit=" records"):
        raw_content = record.get("raw_content")
        if not raw_content: 
            duplicates_found +=1 
            continue
            
        content_hash = hashlib.sha256(raw_content.encode('utf-8')).hexdigest()
        if content_hash not in unique_contents:
            unique_contents[content_hash] = record
        else:
            duplicates_found += 1

    logger.info(f"  Number of duplicate records found and removed: {duplicates_found}")
    logger.info(f"  Number of unique records after de-duplication: {len(unique_contents)}")

    # Characterization and writing to final output file
    logger.info(f"  Characterizing unique records and writing to {FINAL_OUTPUT_JSONL}...")
    final_written_count = 0
    with open(FINAL_OUTPUT_JSONL, "w", encoding="utf-8") as outfile:
        for content_hash, record in tqdm(unique_contents.items(), desc="  Characterizing & Writing", unit=" records"):
            language = record.get("language", "unknown").lower() 
            raw_content = record.get("raw_content")

            if not raw_content: 
                continue

            char_func = get_characterization_function(language)
            characteristics = char_func(raw_content)
            
            record["snippet_id"] = content_hash 
            record["characteristics"] = characteristics
            
            outfile.write(json.dumps(record) + "\n")
            final_written_count += 1
            
    logger.info(f"  Total unique, characterized records written: {final_written_count}")
    logger.info(f"🎉 Stage 2 complete. Final unlabeled pool at: {FINAL_OUTPUT_JSONL}")
    logger.info(f"Log file for this stage: {LOG_FILE}")

if __name__ == "__main__":
    # ---- DEBUG PRINT STATEMENT ----
    print("DEBUG: Stage 2 script - __main__ block reached, attempting to call main().")
    main()
    print("DEBUG: Stage 2 script - main() function has completed or exited.")
