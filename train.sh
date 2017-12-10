#!/bin/bash

set -o xtrace

xvfb-run -s "-screen 0 1400x900x24" bash
source .env/bin/activate
python run.py --train --load --model_dir saved_model/CNN_action/ --ckpt 82000 \
  --batchSize=256 \
  --batchPerFeedback=128 \
  --updateInterval=100 \
  --updateTargetInterval=10

set +o xtrace
