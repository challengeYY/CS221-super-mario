
class QLearningAlgorithm():
    def __init__(self, options, actions, discount, featureExtractor, model, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.numIters = 0
        self.batc, hsize = 10 # number of frames to retrain the model
        self.windowsize = 3  # number of frames to look back in a state
        self.statecache = []
        self.rewardcache = []
        self.actioncache = []
        self.batchcounter = 0
        self.model = model
        self.options = options

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        window = self.statecache[-self.windowsize+1:-1] + [state]
        x = self.featureExtractor((window, action))
        score, = self.model.predict([x])
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if (random.random() < self.explorationProb or len(self.statecache) < self.windowsize) and self.options.isTrain:
            return random.choice(self.actions(state))
        else:
            Qopt = self.getQ(state, action)
            return max((Qopt, action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        if newState is None: return
        reward = get_reward(newState) - get_reward(state)
        self.statecache.append(state)
        self.actioncache.append(action)
        self.rewardcache.append(reward)
        eta = self.getStepSize()

        self.batchcounter += 1
        
        if self.batchcounter >= self.batchsize:
            self.batchcounter = 0
            gamma = self.discount

            X = []
            Y = []
            for i in range(1, self.batchsize+1):
                window = self.statecache[-self.windowsize-i:-i]
                X.append(self.featureExtractor((window, action)))
                reward = self.rewardcache[-i]
                Vopt = max([self.getQ(self.statecache[-self.windowsize:], a) for a in self.actions(newState)])
                target = (reward + gamma * Vopt)
                Y.append(target)
            self.model.train(X, Y)