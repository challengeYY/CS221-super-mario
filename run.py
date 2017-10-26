import gym
import ppaquette_gym_super_mario
from pynput import keyboard
# import gym_pull
# gym_pull.pull('github.com/ppaquette/gym-super-mario')
env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
env.reset()
total_score = 0
action = [0,0,0,0,0,0]
def on_press(key):
    try:
       if key.char == 'a':
           action[1] = 1
       elif key.char == 'd':
           action[3] = 1
       elif key.char == 'j':
           action[4] = 1
       elif key.char == 'k':
           action[5] = 1
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    if key.char == 'a':
        action[1] = 0
    elif key.char == 'd':
        action[3] = 0
    elif key.char == 'j':
        action[4] = 0
    elif key.char == 'k':
        action[5] = 0

# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    while total_score < 32000:
        obs, reward, is_finished, info = env.step(action)
        env.render()
        total_score = info["distance"]
    listener.join()
