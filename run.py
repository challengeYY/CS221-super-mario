import gym
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario.wrappers.control import *
import argparse
from bot import *

def load_options(options):
    print('Loading options ...')
    new_options = options
    option_path = options.model_dir + '/options.pickle'
    if not os.path.isfile(option_path):
        print('No parameters stored in {}'.format(option_path))
        exit(-1)
    options = pickle.load(open(option_path, 'rb'))
    options.isTrain = new_options.isTrain
    options.model_dir = new_options.model_dir
    options.ckpt = new_options.ckpt
    options.maxGameIter = new_options.maxGameIter
    options.batchSize = new_options.batchSize
    options.batchPerFeedback = new_options.batchPerFeedback
    options.updateInterval = new_options.updateInterval
    options.updateTargetInterval = new_options.updateTargetInterval
    options.save_period = new_options.save_period

    if not hasattr(options, 'conv_model'):
        options.conv_model = 0

    if not hasattr(options, 'partial_reward'):
        options.partial_reward = False

    if not hasattr(options, 'fix_exprate'):
        options.fix_exprate = False

    if not hasattr(options, 'dist_reward_only'):
        options.dist_reward_only = False

    if options.ckpt < 0:
        for d in [x for x in os.listdir(options.model_dir)]:
            if 'ckpt' in d:
                ckpt = int(d.split('ckpt')[1])
                if options.ckpt < ckpt:
                    options.ckpt = ckpt
        if options.ckpt < 0:
            print('No ckpt in {}!'.format(options.model_dir))
            exit(-1)
        else:
            print('Loading latest ckpt{}...'.format(options.ckpt))

    return options

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
    parser.add_argument('--restore', dest='restore', action='store_true', default=False,
                        help='Restore training')
    parser.add_argument('--ckpt', dest='ckpt', nargs='?', default=0, type=int, help='ckpt number of a training')

    # Game hyper parameter
    parser.add_argument('--maxGameIter', dest='maxGameIter', nargs='?', default=1, type=int,
                        help='Max number of training iteration')
    parser.add_argument('--stepCounterMax', dest='stepCounterMax', nargs='?', default=10, type=int,
                        help='Number of frames to advance state')
    parser.add_argument('--updateInterval', dest='updateInterval', nargs='?', default=10, type=int,
                        help='Number of frames to retrain the model')
    parser.add_argument('--updateTargetInterval', dest='updateTargetInterval', nargs='?', default=50, type=int,
                        help='Number of updates to update the target network')
    parser.add_argument('--maxCache', dest='maxCache', nargs='?', default=1000, type=int,
                        help='Max number of training iteration')
    parser.add_argument('--softmaxExploration', dest='softmaxExploration', action='store_true', default=False,
                        help='Exploration mode')
    parser.add_argument('--window', dest='windowsize', nargs='?', default=3, type=int,
                        help='Number of frames to include in a state')
    parser.add_argument('--tileWindowSize', dest='tileWindowSize', nargs='?', default=3, type=int,
                        help='Number of frames the TileFeatureExtractor extracts')
    parser.add_argument('--prevActionsSize', dest='prevActionsSize', nargs='?', default=10, type=int,
                        help='Number of previous action to include in states')
    parser.add_argument('--batchSize', dest='batchSize', nargs='?', default=20, type=int,
                        help='Number of samples to train the model')
    parser.add_argument('--batchPerFeedback', dest='batchPerFeedback', nargs='?', default=5, type=int,
                        help='Number of batched updates before continue playing')
    parser.add_argument('--explorationProb', dest='explorationProb', nargs='?', default=0.5,
            type=float, help='Exploration Probability. Decay over time')
    parser.add_argument('--explorationGrowthFactor', dest='explorationGrowthFactor', nargs='?', default=100.0,
                        type=float, help='factor which control the increasing/decreasing rate of exploration prob around center.')
    parser.add_argument('--explorationPercentile', dest='explorationPercentile', nargs='?', default=0.9,
                        type=float, help='top percentile of playback buffer for deciding midpoint of exploration function.')

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

    if options.isTrain and not options.restore:
        options.model_dir = "model/{:%Y%m%d_%H%M%S}".format(datetime.now())
    else: # testing. loading options
        options = load_options(options)

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
    elif options.player == 'feature':
        agent = FeatureAgent(options, env)
    elif options.player == 'manual':
        agent = ManualFeatureAgent(options, env)

    if not os.path.exists(options.model_dir):
        os.makedirs(options.model_dir)

    if not options.isTrain:
        options.maxGameIter = 1

    env.reset()

    try:
        action = agent.initAction()
        while not agent.exit():
            total_reward = 0
            skip_frames = 1
            for i in range(skip_frames):
                obs, reward, is_finished, info = env.step(action)
                total_reward += reward
            if options.render:
                env.render()

            action = agent.act(obs, total_reward - skip_frames, is_finished, info)
    except Exception as e:
        agent.handle(e)

if __name__ == "__main__":
    main()
