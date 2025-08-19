#!/bin/bash

python3.12 -m venv .venv
. .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
