import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.w1 = nn.Parameter(state_dim, 100)
        self.b1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 100)
        self.b2 = nn.Parameter(1, 100)
        self.w3 = nn.Parameter(100, action_dim)
        self.b3 = nn.Parameter(1, action_dim)
        self.parameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        self.learning_rate = -1.5
        self.numTrainingGames = 2000
        self.batch_size = 64

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        predQ = self.run(states)
        loss = nn.SquareLoss(predQ, Q_target)
        return loss


    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        statesW1 = nn.Linear(states, self.w1)
        predQ1 = nn.AddBias(statesW1, self.b1)
        relU1 = nn.ReLU(predQ1)
        statesW2 = nn.Linear(relU1, self.w2)
        predQ2 = nn.AddBias(statesW2, self.b2)
        relU2 = nn.ReLU(predQ2)
        statesW3 = nn.Linear(relU2, self.w3)
        predQ3 = nn.AddBias(statesW3, self.b3)

        return predQ3


    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"

        loss = self.get_loss(states, Q_target)
        gradientss = nn.gradients(loss, self.parameters)
        self.w1.update(gradientss[0], self.learning_rate)
        self.b1.update(gradientss[1], self.learning_rate)
        self.w2.update(gradientss[2], self.learning_rate)
        self.b2.update(gradientss[3], self.learning_rate)
        self.w3.update(gradientss[4], self.learning_rate)
        self.b3.update(gradientss[5], self.learning_rate)
