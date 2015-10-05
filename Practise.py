import numpy

import theano
import theano.tensor as t
import matplotlib.pyplot as plt

__author__ = 'slevin'

if __name__ == '__main__':
    rand = numpy.random

    # Training Data
    X = numpy.asarray([1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83])
    Y = numpy.asarray(
        [52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46])

    # Initialize m and b to a random int
    m_value = rand.random()
    b_value = rand.random()

    # Make m and b shared variables for efficiency
    m = theano.shared(m_value, name='m')
    b = theano.shared(b_value, name='b')

    # Make x and y symbolic variables (for dat speed)
    x = t.vector('x')
    y = t.vector('y')

    # Get the number of elements in x
    num_samples = X.shape[0]

    # Prediction is the predicted line from x numpy array
    prediction = t.dot(x, m) + b

    # Error is the predicted minus actual squared
    error = t.sum(t.pow(prediction - y, 2))

    # Cost is the average error and we are minimizing one half of the function to make the math easier
    cost = error / (2 * num_samples)

    # Compute Gradient
    grad_m = t.grad(cost, m)
    grad_b = t.grad(cost, b)

    # Set learning rate and amount of epochs
    learning_rate = 0.2
    epochs = 1000

    """Train function

    Args:
        x: the numpy array of x values
        y: the numpy array of y values
    Returns:
        cost: the average error
    Updates:
        m: m is updated with a new m. New m is based on learning rate and new grad_m
        b: b is updated with a new b. New b is based on learning rate and new grad_b
    """
    train = theano.function([x, y], cost,
                            updates=[(m, m - (grad_m * learning_rate)), (b, b - (grad_b * learning_rate))])

    """Predict function

    Args:
        x: the numpy array of x values
    Returns:
        prediction: the line created from the trained m and b values
    """
    predict = theano.function([x], prediction)

    # Run for the amount of epochs
    for i in range(epochs):
        epoch_cost = train(X, Y)
        print("Epoch: %s, Average error: %s" % (i, epoch_cost))

    print("Slope: %s" % m.get_value())
    print("Intercept: %s" % b.get_value())

    predict_values = numpy.linspace(1, 2)
    predicted = predict(predict_values)
    plt.plot(X, Y, 'bs')
    plt.plot(predict_values, predicted, 'red')
    plt.show()
