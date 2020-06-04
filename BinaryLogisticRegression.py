from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE      = 0.1   # The learning rate.
    MINIBATCH_SIZE     = 1000  # Minibatch size (only for minibatch gradient descent)
    RATIO              = 0.9   # The split ratio, i.e. what part of training data remains in the training set
    PATIENCE           = 5     # Max number of validation set epochs where loss can increase
    EPSILON            = 0.0001 # Convergence criterion
    GRAD_DIFF          = 0.000000000003
    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """

        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        
        if theta:
            # The model exists
            self.FEATURES = len(theta)
            self.theta = theta
        elif x and y:
            # The model should be trained. First split the data into a training set and 
            # a validation (development) set.
            x_tr, y_tr, x_val, y_val = self.train_val_split(np.array(x), np.array(y), ratio=self.RATIO)

            # Number of training datapoints.
            self.TRAINING_DATAPOINTS = len(x_tr)

            # Number of validation datapoints.
            self.VALIDATION_DATAPOINTS = len(x_val)

            # Number of features.
            self.FEATURES = len(x_tr[0]) + 1

            # Encoding of the training data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.TRAINING_DATAPOINTS, 1)), x_tr), axis=1)

            # Correct labels for the training datapoints.
            self.y = y_tr

            # Encoding of the validation data points (as a DATAPOINTS x FEATURES size array).
            self.x_val = np.concatenate((np.ones((self.VALIDATION_DATAPOINTS, 1)), x_val), axis=1)

            # Correct labels for the validation datapoints.
            self.y_val = y_val

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)

            self.last_loss = 0



    # ----------------------------------------------------------------------

    def train_val_split(self, x, y, ratio):
        """
        Performs a split of the given training data into a training set and a validation set
        
        :param      x:      The input features of the training data
        :param      y:      The correct labels of the training data
        :param      ratio:  The split ratio, i.e. what part of training data remains in the training set
                            e.g. ratio = 0.8 means that 80% of the training data will be used as training set
                                       and the remaining 20% will be used a validation set
        """
        # REPLACE THE COMMAND BELOW WITH YOUR CODE

        full_set = np.arange(len(x))
        np.random.shuffle(full_set)

        training_set = full_set[:int(math.ceil(len(full_set)*ratio))]
        validation_set = full_set[int(math.ceil(len(full_set)*ratio)):]

        training_fset = []
        training_cset = []
        for i in training_set:
            training_fset.append(x[i])
            training_cset.append(y[i])

        validation_fset = []
        validation_cset = []
        for i in validation_set:
            validation_fset.append(x[i])
            validation_cset.append(y[i])
        
        return training_fset,training_cset, validation_fset, validation_cset


    def loss(self, x, y):
        """
        Computes the loss function given the input features x and labels y
        
        :param      x:    The input features
        :param      y:    The correct labels

        x = all data points
        y = all classifications
        """

        # Implements cross-entropy loss function
        summation = 0
        for i in range(len(x)):
            lhs = -y[i] * math.log(self.sigmoid(np.dot(np.transpose(self.theta),x[i])))
            rhs = (1- y[i]) * math.log(1- self.sigmoid(np.dot(np.transpose(self.theta),x[i])))
            summation += lhs - rhs

        return (1/float(len(x)) *  summation)

    def compute_validation_loss(self, val_loss, inc_val_loss):
        """
        Calculates the validation loss, and tracks the number of consecutive iterations
        the validation loss increases, using the `inc_val_loss` variable.
        
        :param      val_loss:      The current value of the validation loss
        :param      inc_val_loss:  The number of iterations of constant increase of validation loss
        """
        new_val_loss = self.loss(self.x_val, self.y_val)
        
        if(new_val_loss > val_loss):
            return new_val_loss, inc_val_loss+1
        else:
            return 0,0

        #return val_loss, inc_val_loss


    def sigmoid(self, z):
        """
        The logistic function at the point z.
        """
        return 1.0 / ( 1 + math.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        
        :param      label:      The label
        :param      datapoint:  The datapoint itself (NOT the ID)
        """
        cond_prob = self.sigmoid(np.dot(np.transpose(self.theta), datapoint))
        if label == 1:
            return cond_prob
        else:
            return 1-cond_prob

    def sum_compute_gradient(self, k, batch=[]):
        summation = 0
        real_batch = range(1,self.TRAINING_DATAPOINTS)
        if len(batch) > 0:
            real_batch = batch
        for i in  real_batch:
            xi = self.x[i]
            yi = self.y[i]
            hypothesis = self.sigmoid(np.dot(np.transpose(self.theta),xi))
            summation += xi[k] * (hypothesis - yi)
        return (1/float(len(real_batch))) * summation
        
    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """
        # trivially parallelizable
        for k in range(self.FEATURES):
            self.gradient[k] = self.sum_compute_gradient(k)


    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for mini-batch gradient descent).

        :param      minibatch:  A list of IDs for the datapoints in the minibatch
        """
        for k in range(self.FEATURES):
            self.gradient[k] = self.sum_compute_gradient(k, batch=minibatch)


    def compute_gradient(self, datapoint):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        
        :param      datapoint:  The ID of the datapoint in which the gradient is to be computed
        """
        for k in range(self.FEATURES):
            hypothesis = self.sigmoid(np.dot(np.transpose(self.theta),self.x[datapoint]))
            self.gradient[k] = self.x[datapoint][k] *  (hypothesis - self.y[datapoint])

    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        print(self.FEATURES)
        #self.init_plot(self.FEATURES)
        ith = random.randint(0,self.TRAINING_DATAPOINTS-1)
        iteration = 1
        val_loss = 0
        inc_val_loss = 0
        early_stop = False
        early_stop_iter = 0

        last_gradient = float("inf")
        print("SArt")
        while self.GRAD_DIFF < abs(last_gradient - np.sum(self.gradient**2)):
        #while True:
            last_gradient = np.sum(self.gradient**2)
            # Early stopping
            if iteration % 1000 == 0:
                val_loss, inc_val_loss = self.compute_validation_loss(val_loss,inc_val_loss)
                early_stop_iter += 1
            if(inc_val_loss > self.PATIENCE):
                print("EARLY STOPPING!")
                return
            if iteration%100 == 0:
                print(iteration)

            iteration = iteration + 1
            #if iteration%100 == 0:
            #self.update_plot(self.loss(self.x,self.y),self.loss(self.x_val, self.y_val))
                

            ith = random.randint(0,self.TRAINING_DATAPOINTS-1)
            self.compute_gradient(ith)
            for k in range(self.FEATURES):
                self.theta[k] = self.theta[k] - self.LEARNING_RATE * self.gradient[k]


    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        #self.init_plot(self.FEATURES)
        iteration = 1
        val_loss = 0
        inc_val_loss = 0

        self.compute_gradient_minibatch(np.random.choice(len(self.x), self.MINIBATCH_SIZE, replace = False))
        for k in range(self.FEATURES):
            self.theta[k] = self.theta[k] - self.LEARNING_RATE * self.gradient[k]
        last_gradient = float("inf")
        while self.GRAD_DIFF < abs(last_gradient - np.sum(self.gradient**2)):
            if iteration%100 == 0:
                print(iteration)
            if iteration%100 == 0:
                last_gradient = np.sum(self.gradient**2)
            if iteration%1000 == 0:    
                val_loss, inc_val_loss = self.compute_validation_loss(val_loss,inc_val_loss)
            if(inc_val_loss > self.PATIENCE):
                print("EARLY STOPPING!")
                return

            
            self.compute_gradient_minibatch(np.random.choice(len(self.x), self.MINIBATCH_SIZE, replace = False))
            for k in range(self.FEATURES):
                self.theta[k] = self.theta[k] - self.LEARNING_RATE * self.gradient[k]
                
            iteration = iteration + 1
            #if iteration%10 == 0:
            #    self.update_plot(self.loss(self.x,self.y),self.loss(self.x_val, self.y_val))
        # YOUR CODE HERE


    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        #self.init_plot(self.FEATURES)
        i = 1
        val_loss = 0
        inc_val_loss = 0

        self.compute_gradient_for_all()
        for k in range(self.FEATURES):
            self.theta[k] = self.theta[k] - self.LEARNING_RATE * self.gradient[k]
        
        while self.EPSILON < np.sum(self.gradient**2):
            print(i)
            val_loss, inc_val_loss = self.compute_validation_loss(val_loss,inc_val_loss)
            if(inc_val_loss > self.PATIENCE):
                print("EARLY STOPPING!")
                return
            
            self.compute_gradient_for_all()
            for k in range(self.FEATURES):
                self.theta[k] = self.theta[k] - self.LEARNING_RATE * self.gradient[k]

            i = i + 1
            #if i%10 == 0:
            #    self.update_plot(self.loss(self.x,self.y),self.loss(self.x_val, self.y_val))
            


    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        
        :param      test_data:    The input features for the test set
        :param      test_labels:  The correct labels for the test set
        """
        print('Model parameters:')

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        num_test_datapoints = len(test_data)

        x_test = np.concatenate((np.ones((num_test_datapoints, 1)), np.array(test_data)), axis=1)
        y_test = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(num_test_datapoints):
            prob = self.conditional_prob(1, x_test[d])
            predicted = 1 if prob > .5 else 0
            confusion[predicted][y_test[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        Initializes the plot.
        
        :param      num_axes:  The number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        colors = [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=colors[i], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    b.fit()
    b.print_result()


if __name__ == '__main__':
    main()