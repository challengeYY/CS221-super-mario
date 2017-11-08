
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

class Action:
    NAME = ['Up', 'Left', 'Down', 'Right', 'A', 'B']

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
        action = [0] * len(Action.NAME)
        if type(name) is list: # a list of names
            for n in name:
                action[Action.index(n)] = 1
        else: # a single name
            action[Action.index(name)] = 1
        return action
