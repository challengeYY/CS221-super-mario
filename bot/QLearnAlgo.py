import random
from util import *

class QLearningAlgorithm():
    def __init__(self, options, actions, discount, featureExtractor, windowsize, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.batchsize = 10 # number of frames to retrain the model
        self.windowsize = windowsize  # number of frames to look back in a state
        self.statecache = []
        self.actioncache = []
        self.batchcounter = 0
        self.options = options
        self.model = None

    def set_model(self, model):
        self.model = model

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        window = self.statecache[-self.windowsize:-1] + [state]
        x = self.featureExtractor(window, action)
        score, = self.model.inference([x])
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        rand = random.random()
        if rand < self.explorationProb and self.options.isTrain:
            actionName = random.choice(self.actions(state))
        elif len(self.statecache) < self.windowsize:
            actionName = 'Right'
        else:
            actionName = max((self.getQ(state, action), action) for action in self.actions(state))[1]
        return Action.act(actionName)

    # Call this function to get the step size to update the weights.

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        self.batchcounter += 1

        if self.batchcounter >= self.batchsize:
            self.batchcounter = 0
            gamma = self.discount

            X = []
            Y = []
            for i in range(1, self.batchsize+1):
                window = self.statecache[-self.windowsize-i-1:-i]
                X.append(self.featureExtractor((window, action)))
                reward = get_reward(self.statecache[-i])
                Vopt = max([self.getQ(self.statecache[-self.windowsize:], a) for a in self.actions(newState)])
                target = (reward + gamma * Vopt)
                Y.append(target)
            self.model.update_weights(X, Y)
