#!/bin/bash

set -o xtrace

source .env/bin/activate

#python run.py --load --model_dir saved_model/CNN_action/ --ckpt 82000
#python run.py --train --load --model_dir saved_model/CNN_action/ --ckpt 82000 \
  #--maxGameIter=3000 \
  #--batchSize=256 \
  #--batchPerFeedback=128 \
  #--updateInterval=100 \
  #--updateTargetInterval=10 \
  #--save_period=1000

#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--batchSize=256 \
  #--batchPerFeedback=128 \
  #--updateInterval=100 \
  #--updateTargetInterval=10 \
  #--save_period=1000 \
  #--conv_model=1 \
  #--load --model_dir saved_model/CNN_action/ --ckpt 82000 \
  #--maxGameIter=3000 \

./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  --batchSize=512 \
  --batchPerFeedback=128 \
  --updateInterval=100 \
  --updateTargetInterval=10 \
  --save_period=1000 \
  --conv_model=1 \
  --lr=0.001 \
  --load --model_dir saved_model/CNN_action/ --ckpt 82000 \
  --maxGameIter=3000 \
set +o xtrace
