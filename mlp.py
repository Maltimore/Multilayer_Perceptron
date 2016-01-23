import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(x))
    
def deriv_sigmoid(x):
    return np.multiply(x, (1-x))

def construct_network(n_out, n_hidden=6):
    W1 = np.matrix(np.random.normal(size=(n_hidden, x.shape[0]+1)))
    W2 = np.matrix(np.random.normal(size=(n_out, n_hidden+1)))
    return W1, W2

def forward_propagate(x, W1, W2):
    hidden = sigmoid(W1 @ x)
    hidden = np.vstack((hidden, np.matrix([1]))) # adding the bias
    output = W2 @ hidden
    return hidden, output

def classify(X, W1, W2):
    Y_hat= np.matrix(np.empty((n_data)))
    for data_idx, x in enumerate(X.T):
        x = X[:, data_idx]
        y = Y[0, data_idx]
        _, output = forward_propagate(x, W1, W2)
        Y_hat[:, data_idx] = output
    return Y_hat

def compute_error(Y, Y_hat, n):
    return (Y_hat - Y) @ (Y_hat - Y).T / n

# create sample data
n_data = 100
n_loops = 10000
x_dimension = 1
n_hidden = 20
X = np.matrix((np.random.normal(size=(x_dimension, n_data))))
X = np.vstack((X, np.matrix(np.ones(X.shape[1]))))
Y = np.matrix(np.empty(X.shape[1]))
for idx, x in enumerate(X.T):
    Y[0,idx] = x[0,0]**2 # + np.random.normal(scale=.5)
W1, W2 = construct_network(Y.shape[0], n_hidden)

Y_hat = classify(X, W1, W2)
error = compute_error(Y, Y_hat, n_data)
plt.figure()
plt.plot(X[0,:].T, Y[0,:].T, "x", label="true")
plt.plot(X[0,:].T, Y_hat[0,:].T, "x", label="predicted")
plt.legend()
plt.title("Error is: " + str(error))

errorvec = np.empty((n_loops))
for loop in np.arange(n_loops):
    delta_W1 = np.zeros(W1.shape)
    delta_W2 = np.zeros(W2.shape)
    for data_idx in np.arange(n_data):
        x = X[:, data_idx]
        y = Y[:, data_idx]
        hidden, output = forward_propagate(x, W1, W2)
        
        delta_output = output - y
        #print("delta_output is: " + str(delta_output))
        delta_hidden = -np.multiply(deriv_sigmoid(hidden), W2.T @ delta_output)
        delta_hidden = delta_hidden[:-1,0] # removing bias
        delta_W2 -= delta_output @ hidden.T
        delta_W2[:,:-1] = np.mean(delta_hidden)
        delta_W1 -= delta_hidden @ x.T
        delta_W1[:,:-1] = np.mean(x[:-1,0])

    W1 += 1/n_data * delta_W1
    W2 += 1/n_data * delta_W2
    
    Y_hat = classify(X, W1, W2)
    error = compute_error(Y, Y_hat, n_data)
    errorvec[loop] = error

plt.figure()
plt.plot(np.arange(n_loops), errorvec)

Y_hat = classify(X, W1, W2)
plt.figure()
plt.plot(X[0,:].T, Y[0,:].T, "x", label="true")
plt.plot(X[0,:].T, Y_hat[0,:].T, "x", label="predicted")
plt.legend()
plt.title("Error is: " + str(error))