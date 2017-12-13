#!/bin/bash

set -o xtrace

source .env/bin/activate

# tucson
#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--maxGameIter=3000 \
  #--load --model_dir saved_model/CNN_action/ --ckpt=154000 \
#python tools/plot_scores.py --score_log_path saved_model/CNN_action/score_log 

#./xvfb-run-safe -s "-screen 1 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--maxGameIter=3000 \
  #--load --model_dir model/20171210_154650/ --ckpt=126000
#python tools/plot_scores.py --score_log_path model/20171210_154650/score_log 

#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--maxGameIter=3000 \
  #--batchSize=64 \
  #--batchPerFeedback=16 \
  #--updateInterval=64 \
  #--load --model_dir model/20171212_181553/ --ckpt=11392
#python tools/plot_scores.py --score_log_path model/20171212_181553/score_log 

#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--maxGameIter=3000 \
  #--batchSize=128 \
  #--batchPerFeedback=16 \
  #--updateInterval=256 \
  #--save_period=500 \
  #--load --model_dir model/20171212_210533/ --ckpt 864
#python tools/plot_scores.py --score_log_path model/20171212_210533/score_log 

#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--maxGameIter=3000 \
  #--load --model_dir model/20171212_182429/ --ckpt 52000
#python tools/plot_scores.py --score_log_path model/20171212_182429/score_log 

#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--maxGameIter=3000 \
  #--lr=1e-3 \
  #--load --model_dir model/20171212_182700/ --ckpt=60000
#python tools/plot_scores.py --score_log_path model/20171212_182700/score_log 

#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--maxGameIter=3000 \
  #--lr=1e-2 \
  #--load --model_dir model/20171212_182911/ --ckpt=14000
#python tools/plot_scores.py --score_log_path model/20171212_182911/score_log 

#./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
  #--conv_model=1 \
  #--partial_reward \
  #--maxGameIter=3000 \
  #--load --model_dir model/20171213_021909/ --ckpt 0
#python tools/plot_scores.py --score_log_path model/20171213_021909/score_log 

set +o xtrace
