from pynput import keyboard
from abc import ABCMeta, abstractmethod
import traceback
from controller import *

class HumanController(Controller):
    def __init__(self, options):
        self.action = [0,0,0,0,0,0]
        self.state = None
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release) as listener:
                self.listener = listener

    def initAction(self):
        print('init act ...')
        return self.action

    def act(self, obs, reward, is_finished, info):
        self.state = (obs, reward, is_finished, info)
        print('acting {}'.format(self.action))
        return self.action

    def exit(self):
        if self.state is None:
            return False
        (obs, reward, is_finished, info) = self.state
        total_score = info["distance"]
        stop = total_score > 32000
        if is_finished or stop:
            self.listener.join()
            if stop:
                print('exiting the game cuz I no longer wanna play ...')
            return True
        else:
            return False

    def on_press(self, key):
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
    
    def on_release(self, key):
        if key.char == 'a':
            action[1] = 0
        elif key.char == 'd':
            action[3] = 0
        elif key.char == 'j':
            action[4] = 0
        elif key.char == 'k':
            action[5] = 0

    def handle(self, e):
        self.listener.join()
        print('encountering error, exiting ...')
        traceback.print_exc()
        exit(-1)

