from pynput import keyboard
from abc import ABCMeta, abstractmethod
import traceback


# Agent interface to interact with emulator
class Agent:
    __metaclass__ = ABCMeta

    # Agent's initial action 
    @abstractmethod
    def initAction(self): pass

    # Agent returns a action based on the feedback from emulator
    @abstractmethod
    def act(self, obs, reward, is_finished, info): pass

    # Agent can determine when does it want to stop playing the game
    @abstractmethod
    def exit(self): pass

    # Error handling 
    @abstractmethod
    def handle(self, e): pass
