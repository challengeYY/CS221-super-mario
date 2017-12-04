rm ./train/*
python run.py --player=feature --train --window=10 --maxGameIter=100000 --softmaxExploration

