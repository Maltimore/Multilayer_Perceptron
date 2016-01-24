import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(x))
    
def deriv_sigmoid(x):
    return np.multiply(x, (1-x))

def construct_network(layer_sizes):
    weights = []
    for idx in range(len(layer_sizes)-1):
        weights.append(np.matrix(np.random.normal(size=(layer_sizes[idx+1],layer_sizes[idx]+1))))
    return weights

def forward_propagate(x, weights):
    layer_activations = [np.vstack((x, np.matrix([1])))]
    for idx in range(len(weights)-1):
        W = weights[idx]
        layer_activations.append(sigmoid(W @ layer_activations[idx]))
        layer_activations[idx+1] = np.vstack((layer_activations[idx+1], np.matrix([1]))) # adding the bias
    output = weights[-1] @ layer_activations[-1]
    layer_activations.append(output)    
    return layer_activations, output

def classify(X, weights):
    Y_hat= np.matrix(np.empty((n_data)))
    for data_idx, x in enumerate(X.T):
        x = X[:, data_idx]
        _, output = forward_propagate(x, weights)
        Y_hat[:, data_idx] = output
    return Y_hat

def compute_error(Y, Y_hat, n):
    return (Y_hat - Y) @ (Y_hat - Y).T / n

# create sample data
n_data = 100
n_loops = 200
x_dimension = 1
X = np.matrix((np.random.normal(size=(x_dimension, n_data))))
Y = np.matrix(np.empty(X.shape[1]))

#def train(X, Y, hidden_layer_sizes, error_deriv):
hidden_layer_sizes = [30, 30] # DELETE LATER
layer_sizes = [X.shape[0]] + hidden_layer_sizes + [Y.shape[0]]
for idx, x in enumerate(X.T):
    Y[0,idx] = x[0,0]**2 # + np.random.normal(scale=.5)
weights = construct_network(layer_sizes)

errorvec = np.empty((n_loops))
for loop in np.arange(n_loops):
    delta_W = []
    # initiazlizing delta Ws
    for weight_mat_idx in range(len(weights)):
        delta_W.append(np.zeros((weights[weight_mat_idx].shape)))
        
    for data_idx in np.arange(n_data):
        x = X[:, data_idx]
        y = Y[:, data_idx]
        layer_activations, output = forward_propagate(x, weights)
        
        errors = [output - y]
        # go from the output layer towards the input layer and calculate error values
        for idx in np.arange(1, len(weights)):
            # prepend the newest error layer to the errors list:
            errors = [-np.multiply(deriv_sigmoid(layer_activations[-idx-1]),  \
                                   weights[-idx].T @ errors[-idx])]             \
                     + errors
            errors[-idx-1] = errors[-idx-1][:-1,0] # removing bias
        # go from input layer towards output layer and calculate weight updates
        for idx in range(len(weights)):
            delta_W[idx] -= errors[idx] @ layer_activations[idx].T
            delta_W[idx][:,:-1] = np.mean(layer_activations[idx])
    
    # update all weight matrices
    for idx in range(len(weights)):    
        weights[idx] += 1/n_data * delta_W[idx]
    
    Y_hat = classify(X, weights)
    error = compute_error(Y, Y_hat, n_data)
    errorvec[loop] = error

plt.figure()
plt.plot(np.arange(n_loops), errorvec)

Y_hat = classify(X, weights)
plt.figure()
plt.plot(X[0,:].T, Y[0,:].T, "x", label="true")
plt.plot(X[0,:].T, Y_hat[0,:].T, "x", label="predicted")
plt.legend()
plt.title("Error is: " + str(error))