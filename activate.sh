#!/bin/bash
set -e

echo "🚀 Starting environment setup..."

# Redirect to logfile
LOGFILE="setup_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

# Update and install system packages
sudo apt update && sudo apt install -y git python3-pip python3-venv

# Configure Git identity
git config --global user.name "j-mazz"
git config --global user.email "jrogue.mazz@gmail.com"

# Create and activate virtual environment
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate

pip install -r REQUIREMENTS.txt
# Install remaining dependencies
pip install transformers datasets pandas numpy torch torch_xla tqdm

# Run preprocessing scripts
echo "⚙️ Running optimized_stage1_preprocess.py..."
python preprocess.py

echo "⚙️ Running stage2_combine_clean.py..."
stage2_characterize_dedup.py

# Run training
echo "⚙️ Running training script..."
stage3_active_train_tpu.py

echo "✅ All stages complete."
