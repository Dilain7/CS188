import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        dataPoint = x

        # Compute the dot product between the input factors and the weight vector
        scorePerceptron = nn.DotProduct(dataPoint, self.w)


        #Don't I need to sum together all the elements of the dot product


        return scorePerceptron


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"

        # Calcualte the dotproduct between the input factors and the weight vector
        dotProdNode = nn.DotProduct(x, self.w)
        scaleDotProd = nn.as_scalar(dotProdNode)

        if scaleDotProd >= 0:
            return 1
        else:
            return -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"

        dataSet = dataset

        while True:
            complete = True
            batchSize = 1
            for x, y in dataSet.iterate_once(batchSize):
                realY = nn.as_scalar(y)
                predY = self.get_prediction(x)
                if realY != predY:
                    complete = False
                    self.w.update(x, realY)
            if complete:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        self.w1 = nn.Parameter(1, 20)
        self.b1 = nn.Parameter(1, 20)
        self.w2 = nn.Parameter(20, 1)
        self.b2 = nn.Parameter(1, 1)
        self.parameters = [self.w1, self.b1,  self.w2, self.b2]
        self.learnRate = -0.02


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        xW1 = nn.Linear(x, self.w1)
        predictedY1 = nn.AddBias(xW1, self.b1)
        relU1 = nn.ReLU(predictedY1)
        xW2 = nn.Linear(relU1, self.w2)
        predictedY2 = nn.AddBias(xW2, self.b2)
        return predictedY2



    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictedY = self.run(x)
        loss = nn.SquareLoss(predictedY, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batchSize = 4
        while True:
            for x,y in dataset.iterate_once(batchSize):
                loss = self.get_loss(x, y)
                gradientss = nn.gradients(loss, self.parameters)
                self.w1.update(gradientss[0], self.learnRate)
                self.b1.update(gradientss[1], self.learnRate)
                self.w2.update(gradientss[2], self.learnRate)
                self.b2.update(gradientss[3], self.learnRate)
            if nn.as_scalar(loss) <= 0.002:
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        self.w1 = nn.Parameter(784, 100)
        self.b1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 10)
        self.b2 = nn.Parameter(1,10)
        self.parameters = [self.w1, self.b1, self.w2, self.b2]
        self.learnRate = -0.02

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        xW1 = nn.Linear(x, self.w1)
        predictedY1 = nn.AddBias(xW1, self.b1)
        relU1 = nn.ReLU(predictedY1)
        xW2 = nn.Linear(relU1, self.w2)
        predictedY2 = nn.AddBias(xW2, self.b2)
        return predictedY2


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        fx = self.run(x)
        return nn.SoftmaxLoss(fx, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batchSize = 4
        while True:
            for x, y in dataset.iterate_once(batchSize):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, self.parameters)
                self.w1.update(gradients[0], self.learnRate)
                self.b1.update(gradients[1], self.learnRate)
                self.w2.update(gradients[2], self.learnRate)
                self.b2.update(gradients[3], self.learnRate)
            if dataset.get_validation_accuracy() > 0.975:
                break