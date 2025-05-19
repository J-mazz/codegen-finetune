#!/bin/bash

echo "🚀 Starting environment setup..."

# Update and install system packages
sudo apt update && sudo apt install -y git python3-pip python3-venv

# Configure Git identity
git config --global user.name "j-mazz"
git config --global user.email "jrogue.mazz@gmail.com"

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install Python dependencies
pip install --upgrade pip
pip install transformers datasets pandas accelerate

# Run preprocessing scripts
echo "⚙️ Running optimized_stage1_preprocess.py..."
python preprocess.py

echo "⚙️ Running stage2_combine_clean.py..."
python stage2_combine_clean.py

# Optional: Run training (uncomment if train script exists and is ready)
echo "⚙️ Running training script..."
python train_codegen.py

echo "✅ All stages complete."
