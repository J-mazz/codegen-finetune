CodeGen Fine-Tuning Pipeline

Fine-tune Salesforce's CodeGen models on custom code datasets using Hugging Face Transformers and DeepSpeed. This project provides a modular, scalable pipeline to train large-scale code generation models efficiently — built by a CS undergrad and self-taught ML engineer.

Features

•	CodeGen Support: Fine-tunes models like `Salesforce/codegen-6B-mono` for generative programming tasks.

•	DeepSpeed Integration: Enables memory-efficient training of large models on limited compute.

•	Custom Dataset Handling: Compatible with JSONL-formatted datasets for fine-grained code modeling.

•	Scripted Pipeline: Includes utilities for preprocessing, training, and checkpointing.

Project Structure

codegen-finetune/

├── data/

│   └── your_dataset.jsonl

├── scripts/

│   ├── train.py

│   └── deepspeed_config.json

├── utils/

│   └── data_preparation.py

├── models/

│   └── checkpoints/

├── requirements.txt

└── README.md

Installation

```bash

git clone https://github.com/J-mazz/codegen-finetune.git

cd codegen-finetune

python -m venv venv && source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

```

Dataset Format

Each line in your dataset should be a JSON object with a `text` field containing the code snippet:



```json

{"text": "def add(a, b):\n    return a + b"}

{"text": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)"}

```

Training

Example (with DeepSpeed):

```bash

deepspeed scripts/train.py \

    --model_name_or_path Salesforce/codegen-6B-mono \

    --train_file data/your_dataset.jsonl \

    --output_dir models/checkpoints \

    --per_device_train_batch_size 1 \

    --gradient_accumulation_steps 16 \

    --num_train_epochs 3 \

    --learning_rate 5e-5 \

    --block_size 1024 \

    --logging_steps 100 \

    --save_steps 500 \

    --deepspeed scripts/deepspeed_config.json \

    --fp16

```

Inference

```python

from transformers import AutoTokenizer, AutoModelForCausalLM



model = AutoModelForCausalLM.from_pretrained("models/checkpoints")

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-6B-mono")



prompt = "def fibonacci(n):"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_length=100)

print(tokenizer.decode(outputs[0]))

```

Evaluation

Evaluate model output using BLEU, ROUGE, or CodeBLEU depending on your use case. Add evaluation metrics or visualizations as needed.

License

MIT License

