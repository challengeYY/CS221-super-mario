# Super Mario Bro AI with Deep Q Learning
#### **Developed using Gym Super Mario environment bundle**
Game play recording: <https://youtu.be/p8vFNvz5ggA>

## How to run
### Requirement
Ubuntu 16.0.4 or higher
If you are using Mac, you need to manually install Fceux from homebrew, and skip that part from setup.sh.
### Setup
```
setup.sh
source .env/bin/activate
```

### Training
```
python run.py --player=feature --maxGameIter=100000 --train --tileWindowSize=3 --prevActionSize=10 --stepCounterMax=3 --updateInterval=100 --batchPerFeedback=200 --batchSize=100
```

### Testing

```
python run.py --model_dir=./model/example
```
