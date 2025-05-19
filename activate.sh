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

# Upgrade pip and install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run preprocessing scripts
echo "⚙️ Running stage1_preprocess.py..."
python preprocess.py

echo "⚙️ Running stage2_combine_clean.py..."
python stage2_combine_clean.py

# Run training
echo "⚙️ Running training script..."
python train_codegen.py

echo "✅ All stages complete."
