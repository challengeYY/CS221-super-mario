from pynput import keyboard
from abc import ABCMeta, abstractmethod
import traceback

class Controller:
    __metaclass__ = ABCMeta

    # Controller's initial action 
    @abstractmethod
    def initAction(self): pass

    # Controller returns a action based on the feedback from simulator 
    @abstractmethod
    def act(self, obs, reward, is_finished, info): pass

    # Controller can determine when does it want to stop playing the game
    @abstractmethod
    def exit(self): pass

    # Error handling 
    @abstractmethod
    def handle(self, e): pass
