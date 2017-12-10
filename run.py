import gym
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario.wrappers.control import *
import argparse
from bot import *

def load_options(options):
    model_dir = options.model_dir
    option_path = options.model_dir + '/options.pickle'
    ckpt = options.ckpt
    load = options.load
    isTrain = options.isTrain
    maxGameIter = options.maxGameIter
    if not os.path.isfile(option_path):
        print('No parameters stored in {}'.format(option_path))
        exit(-1)
    options = pickle.load(open(option_path, 'rb'))
    options.load = load
    options.isTrain = isTrain
    options.model_dir = model_dir
    options.ckpt = ckpt
    options.maxGameIter = maxGameIter
    print('Loading options ...')
    optionDict = vars(options)
    for k in optionDict:
        print(k + ' = ' + str(optionDict[k]))
    return options

def create_agent(options, env):
    if options.player == 'human':
        agent = HumanAgent(options)
        wrapper = SetPlayingMode('human')
        env = wrapper(env)
        options.isTrain = False
    elif options.player == 'baseline':
        agent = BaselineAgent(options)
        wrapper = SetPlayingMode('algo')
        env = wrapper(env)
        options.isTrain = False
    elif options.player == 'cnn':
        agent = CNNFeatureAgent(options, env)
        wrapper = SetPlayingMode('algo')
        env = wrapper(env)
    elif options.player == 'cnnidx':
        agent = CNNActionIndexFeatureAgent(options, env)
        wrapper = SetPlayingMode('algo')
        env = wrapper(env)
    elif options.player == 'feature':
        agent = FeatureAgent(options, env)
        wrapper = SetPlayingMode('algo')
        env = wrapper(env)
    elif options.player == 'manual':
        agent = ManualFeatureAgent(options, env)
        wrapper = SetPlayingMode('algo')
        env = wrapper(env)
    return agent,env

def main():
    usage = "Usage: run [options]"
    parser = argparse.ArgumentParser()

    # Game options 
    parser.add_argument('--model_dir', dest='model_dir', action='store', default='./model',
            help='Directory to store weights')
    parser.add_argument('--player', dest='player', action='store', default='human',
                        help='Specify the player, valid option: human, baseline, feature')
    parser.add_argument('--no-gui', dest='render', action='store_false', default=True,
                        help='Do not render visualization of the game')
    parser.add_argument('--train', dest='isTrain', action='store_true', default=False,
                        help='Training mode')
    parser.add_argument('--load', dest='load', action='store_true', default=False,
                        help='load weights')
    parser.add_argument('--ckpt', dest='ckpt', nargs='?', default=0, type=int, help='ckpt number of a training')

    # Game hyper parameter
    parser.add_argument('--maxGameIter', dest='maxGameIter', nargs='?', default=1, type=int,
                        help='Max number of training iteration')
    parser.add_argument('--stepCounterMax', dest='stepCounterMax', nargs='?', default=6, type=int,
                        help='Number of frames to advance state')
    parser.add_argument('--updateInterval', dest='updateInterval', nargs='?', default=10, type=int,
                        help='Number of frames to retrain the model')
    parser.add_argument('--updateTargetInterval', dest='updateTargetInterval', nargs='?', default=20, type=int,
                        help='Number of updates to update the target network')
    parser.add_argument('--maxCache', dest='maxCache', nargs='?', default=1000, type=int,
                        help='Max number of training iteration')
    parser.add_argument('--softmaxExploration', dest='softmaxExploration', action='store_true', default=False,
                        help='Exploration mode')
    parser.add_argument('--window', dest='windowsize', nargs='?', default=3, type=int,
                        help='Number of frames to include in a state')
    parser.add_argument('--tileWindowSize', dest='tileWindowSize', nargs='?', default=3, type=int,
                        help='Number of frames the TileFeatureExtractor extracts')
    parser.add_argument('--prevActionsSize', dest='prevActionsSize', nargs='?', default=1, type=int,
                        help='Number of previous action to include in states')
    parser.add_argument('--batchSize', dest='batchSize', nargs='?', default=20, type=int,
                        help='Number of samples to train the model')
    parser.add_argument('--batchPerFeedback', dest='batchPerFeedback', nargs='?', default=11, type=int,
                        help='Number of batched updates before continue playing')
    parser.add_argument('--explorationProb', dest='explorationProb', nargs='?', default=0.5,
            type=float, help='Exploration Probability. Decay over time')
    parser.add_argument('--death_penalty', dest='death_penalty', nargs='?', default=-100,
            type=int, help='Death penalty to give if gets killed')

    # Model hyper parameters
    parser.add_argument('--optimizer', dest='optimizer', action='store', default='adam', help='SGD optimizer')
    parser.add_argument('--lr', dest='lr', nargs='?', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--decay_step', dest='decay_step', nargs='?', default=1000, type=int,
            help='decay step')
    parser.add_argument('--decay_rate', dest='decay_rate', nargs='?', default=1, type=int,
            help='decay rate')
    parser.add_argument('--regularization', dest='regularization', nargs='?', default=0.001, type=float,
            help='regularization strength')
    parser.add_argument('--gradient_clip', dest='gradient_clip', nargs='?', default=10, type=int,
            help='gradient_clip')
    parser.add_argument('--save_period', dest='save_period', nargs='?', default=2000, type=int,
            help='save_period')

    (options, args) = parser.parse_known_args()

    env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')

    if options.isTrain and not options.load:
        options.model_dir = "model/{:%Y%m%d_%H%M%S}".format(datetime.now())
    if options.load: # testing. loading options
        options = load_options(options)

    if not os.path.exists(options.model_dir):
        os.makedirs(options.model_dir)

    agent,env = create_agent(options, env)

    env.reset()

    try:
        print('Started game ...')
        action = agent.initAction()
        while not agent.exit():
            obs, reward, is_finished, info = env.step(action)
            if options.render:
                env.render()

            action = agent.act(obs, reward, is_finished, info)
    except Exception as e:
        agent.handle(e)

if __name__ == "__main__":
    main()
