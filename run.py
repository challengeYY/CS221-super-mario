import gym
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario.wrappers.control import *
import argparse
from bot import *

def main():
    usage = "Usage: run [options]"
    parser = argparse.ArgumentParser()
    parser.add_argument('--player', dest='player', action='store', default='human',
            help='Specify the player, valid option: human, baseline')
    parser.add_argument('--no-gui', dest='render', action='store_false',default=True,
        help='Do not render visualization of the game')
    parser.add_argument('--maxIter', dest='maxIter', nargs='?', default=1, type=int,
            help='Max number of training iteration')
    parser.add_argument('--train', dest='isTrain', action='store_true',default=False,
        help='Training mode')
    (options, args) = parser.parse_known_args()

    env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')

    if options.player == 'human':
        agent = HumanAgent(options)
        wrapper = SetPlayingMode('human')
        env = wrapper(env)
    elif options.player == 'baseline':
        agent = BaselineAgent(options)
        wrapper = SetPlayingMode('algo')
        env = wrapper(env)
    elif options.player == 'mfeature':
        agent = FeatureAgent(options, env)

    env.reset()

    try:
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
