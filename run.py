import gym
import ppaquette_gym_super_mario
from time import sleep
import argparse
from bot import *

# import gym_pull
# gym_pull.pull('github.com/ppaquette/gym-super-mario') 

def main():
    usage = "Usage: run [options]"
    parser = argparse.ArgumentParser()
    parser.add_argument('--player', dest='player', action='store', default='human',
            help='Specify the player')
    (options, args) = parser.parse_known_args()

    if options.player == 'human':
        controller = HumanController(options)

    env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')
    env.reset()

    try:
        action = controller.initAction()
        while not controller.exit():
            obs, reward, is_finished, info = env.step(action)
            env.render()
            sleep(1.0/30)
            action = controller.act(obs, reward, is_finished, info)
    except Exception as e:
        controller.handle(e)

if __name__ == "__main__":
    main()
