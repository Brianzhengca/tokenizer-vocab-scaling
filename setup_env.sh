#!/bin/bash
# Create the environment: Python packages plus EMO (OLMo-core) with tied embeddings.
# Run once from the repository root inside a fresh Python 3.12 environment, e.g.
#   conda create -n tokenizer-scaling python=3.12 && conda activate tokenizer-scaling
set -euo pipefail

EMO_COMMIT=246fdec72e599319c717aefe29a1d1fdad7c97a0

python -m pip install -r requirements.txt

if [[ ! -d EMO ]]; then
    git clone https://github.com/allenai/EMO.git
    git -C EMO checkout "$EMO_COMMIT"
    # Adds tie_word_embeddings to the Transformer and fixes gradient accumulation under DDP.
    git -C EMO apply ../emo_tied_embeddings.patch
fi
python -m pip install -e './EMO[wandb]'
