import gym
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario.wrappers.control import *
import argparse
from bot import *


def main():
    usage = "Usage: run [options]"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', dest='model_dir', action='store', default='./model/',
                        help='Directory to store weights')
    parser.add_argument('--player', dest='player', action='store', default='human',
                        help='Specify the player, valid option: human, baseline, feature')
    parser.add_argument('--no-gui', dest='render', action='store_false', default=True,
                        help='Do not render visualization of the game')
    parser.add_argument('--maxGameIter', dest='maxGameIter', nargs='?', default=1, type=int,
                        help='Max number of training iteration')
    parser.add_argument('--maxCache', dest='maxCache', nargs='?', default=1000, type=int,
                        help='Max number of training iteration')
    parser.add_argument('--window', dest='windowsize', nargs='?', default=3, type=int,
                        help='Number of states (including current) used to train')
    parser.add_argument('--train', dest='isTrain', action='store_true', default=False,
                        help='Training mode')
    parser.add_argument('--softmaxExploration', dest='softmaxExploration', action='store_true', default=False,
                        help='Exploration mode')
    (options, args) = parser.parse_known_args()

    env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')

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
