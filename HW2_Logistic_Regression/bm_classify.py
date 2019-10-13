import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.append(b,w)
        X = np.hstack((np.ones(shape=(N,1)),X))

        ys = np.multiply(2,y) - 1
        for iter in range(0,max_iterations):
            P = ys * np.dot(w,X.T)
            P = np.array(P<=0,dtype=int)
            P = ys * P
            average_w = np.dot(P,X)
            w += np.multiply((step_size / N),average_w)




        b = w[0]
        w = w[1:]



        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #


        w = np.append(b,w).reshape((1,D+1))
        X = np.hstack((np.ones(shape=(N,1)),X))
        ys = (np.multiply(-2,y) + 1).reshape((1,N))

        for iter in range(0,max_iterations):
            P = np.dot(w,X.T) * ys
            P = ys * sigmoid(P)
            average_w = np.dot(P,X)
            w = w - (step_size / N) *average_w

        w = w.flatten()
        b = w[0]
        w = w[1:]



        ############################################


    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):

    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1 / (1 + np.exp(- z))
    ############################################

    return value


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic

    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape

    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.dot(w,X.T) + b >= 0
        preds = np.array(preds,dtype=int)

        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.dot(w,X.T) + b >= 0
        preds = np.array(preds,dtype=int)
        ############################################


    else:
        raise "Loss Function is undefined."


    assert preds.shape == (N,)
    return preds


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        # b = b.reshape((C,1))
        # X0 = np.ones((N,1))
        # X = np.hstack((X0,X))
        # W = np.hstack((b,w))
        #
        # for iter in range(0,max_iterations):
        #     idx = np.random.choice(N)
        #     xi = X[idx]
        #     P = np.dot(xi,W.T)
        #     max = np.max(P)
        #     P = np.exp(P - max)
        #     sum = np.sum(P)
        #     P = P / sum
        #     P[y[idx]] -= 1
        #     U = np.dot(P.reshape((C,1)),xi.reshape(1,D+1))
        #     W = W - step_size * U
        #
        # b = W.T[0].flatten()
        # w = W.T[1:].T

        b = b.reshape((C,1))
        W = np.hstack((w,b))
        X0 = np.ones((N,1))
        X = np.hstack((X,X0))
        Y = np.zeros((C,N))
        for i in range(0,N):
            Y[y[i],i] = 1
        for iter in range(0,max_iterations):
            idx = np.random.choice(N)
            Yi = Y.T[idx].reshape(C,1)
            xi = X[idx].reshape((1,D+1))
            P = np.dot(W,xi.T)
            maxs = np.max(P,axis=0)
            P = np.exp(P - maxs)
            sums = np.sum(P,axis=0)
            P = P / sums
            P = P - Yi
            W = W - step_size * np.dot(P,xi)

        b = W.T[-1].flatten()
        w = W.T[:-1].T



        ############################################


    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        b = b.reshape((C,1))
        W = np.hstack((b,w))
        X0 = np.ones((N,1))
        X = np.hstack((X0,X))
        Y = np.zeros((C,N))
        for i in range(0,N):
            Y[y[i],i] = 1
        for iter in range(0,max_iterations):
            P = np.dot(W,X.T)
            maxs = np.max(P,axis=0)
            P = np.exp(P - maxs)
            sums = np.sum(P,axis=0)
            P = P / sums
            P = P - Y
            W = W - step_size / N * np.dot(P,X)

        b = W.T[0].flatten()
        w = W.T[1:].T
        ############################################


    else:
        raise "Type of Gradient Descent is undefined."


    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D
    - b: bias terms of the trained multinomial classifier, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    C = len(w)
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    X = np.hstack((np.ones((N,1)),X))
    b = b.reshape((C,1))
    W = np.hstack((b,w))
    P = np.dot(W,X.T).T
    preds = []
    for i in range(N):
        preds.append(np.argmax(P[i]))
    preds = np.array(preds)
    ############################################

    assert preds.shape == (N,)
    return preds
