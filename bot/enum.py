
class Tile:
    NAME = ['EmptySpace', 'Object', 'Enemy', 'Mario']
    EMPTY_SPACE = 0
    OBJECT = 1
    ENEMY = 2
    MARIO = 3

    @staticmethod
    def name(index):
        return Tile.NAME[index]

class Window:
    Width = 16
    Height = 13

    @staticmethod
    def getFrameSize():
        return Window.Width * Window.Height

class Action:
    NAME = ['Up', 'Left', 'Down', 'Right', 'A', 'B']
    NO_ACTION = 'NO_ACTION'

    @staticmethod
    def index(name):
        return Action.NAME.index(name)

    @staticmethod
    def set(action, name):
        action[Action.index(name)] = 1

    @staticmethod
    def unset(action, name):
        action[Action.index(name)] = 0

    @staticmethod
    def act(name):
        action = Action.empty() 
        if type(name) is list: # a list of names
            for n in name:
                action[Action.index(n)] = 1
        elif name != Action.NO_ACTION: # a single name
            action[Action.index(name)] = 1
        return action

    @staticmethod
    def empty():
        return len(Action.NAME) * [0]

    @staticmethod
    def toString(action):
        n = ""
        for i, act in enumerate(action):
            if act == 1:
                n += Action.NAME[i] + ' '
        if n == "":
            n = Action.NO_ACTION
        return n
