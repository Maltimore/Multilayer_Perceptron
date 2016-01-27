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
        # adding the bias
        layer_activations[idx+1] = np.vstack((layer_activations[idx+1], np.matrix([1])))
    output = weights[-1] @ layer_activations[-1]
    layer_activations.append(output)    
    return layer_activations, output

def classify(X, weights, output_transformation):
    output_dimension = weights[-1].shape[0]
    Y_hat= np.matrix(np.empty((output_dimension, n_data)))
    for data_idx, x in enumerate(X.T):
        x = X[:, data_idx]
        _, output = forward_propagate(x, weights)
        Y_hat[:, data_idx] = output_transformation(output)
    return Y_hat

def train(X, Y=[], hidden_layer_sizes=[], error_deriv="default", n_outputs="default", n_loops=100,
          eta=.1, output_transformation="default", error_function="default"):
    """ This learning the weights of a multilayer perceptron with the backpropagation 
        algorithm.
        Parameters:
        X: (n, m) matrix that holds the m inputs to the MLP. Every column is one input vector
           of length n.
        Y: (k, m) matrix that holds the m target values. Every column is a target vector of
           length k. If a custom error derivative function as well as n_outputs is specified,
           Y does not have to be given as a parameter.
        hidden_layer_sizes: list of pos. integers. Every number in the list specifies the amount of 
            units in the corresponding hidden layer. The amount of numbers in this list
            specifies the amount of hidden layers.
        error_deriv: function that takes as input (output, x, y). output is the output that the
            MLP gave for the current input x. y is the current target vector. If Y is specified,
            on the call to train, the error derivative function has to take y as a parameter,
            otherwise that's not necessary.
        n_outputs: int, number of units in the output layer
        n_loops: int, number of times to loop over the training data
        eta: float, learning rate
    """
    
    # the default error function will be the squared error function, whose derivative is just
    # the output minus the target values
    def square_error_deriv(output, x, y):
        return output - y
    if error_deriv == "default":
        error_deriv = square_error_deriv
    
    # the default error function is the squared error    
    def square_error(Y, Y_hat):
        return np.linalg.norm(Y_hat - Y, ord='fro') / (Y.shape[0]*Y.shape[1])
    if error_function == "default":
        error_function = square_error

    # the default output transformatoin is none
    def none_transformation(output):
        return output
    if output_transformation == "default":
        output_transformation = none_transformation

    if len(Y) > 0:
        if n_outputs == "default":
            n_outputs = Y.shape[0]
    else:
        # setting one y value for the rest of the training
        y = "None"
    layer_sizes = [X.shape[0]] + hidden_layer_sizes + [n_outputs]
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
            
            errors = [error_deriv(output, x, y)]
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
            weights[idx] += eta/n_data * delta_W[idx]
        
        Y_hat = classify(X, weights, output_transformation)
        errorvec[loop] = error_function(Y, Y_hat)
    return weights, errorvec

def GMM_derivative(output, x, y):
    # using notation from the book "Neural Networks for Pattern Recognition" (Bishop, 1995)
    M = 3 # number of gaussians
    c = 1 # dimensionality of target variables
    # extracting output activations
    z_alpha = output[:M,0]
    z_sigma = output[M:2*M,0]
    z_mu = output[2*M:,0]
    # converting output activations to parameters
    alphas = np.exp(z_alpha) / np.sum(np.exp(z_alpha))
    sigmas = np.exp(z_sigma)
    print("sigmas")
    print(sigmas)
    mus = z_mu
    
    # computing phis and pis
    phis = np.matrix(np.empty(M)).T
    for j in range(M):
        phis[j] = 1/(2*np.pi*sigmas[j]) * np.exp(-((y-mus[j])**2) /(2*sigmas[j]))
    pis = np.multiply(alphas, phis) / (np.sum(np.multiply(alphas, phis)))
    print("pis")
    print(pis)
    print(np.sum(pis))
    # error deriv wrt alphas
    alpha_error = alphas - pis
    print("alpha_deriv:")
    print(alpha_error)
    # error deriv wrt sigmas
    sigma_error = -np.multiply(pis, (((y-mus).T @ (y-mus))/np.square(sigmas) - c))
    print("sigma_deriv:")
    print(sigma_error)    
    # error deriv wrt mus
    mu_error = np.multiply(pis, (mus-y)/np.square(sigmas))
    return np.vstack((alpha_error, sigma_error, mu_error))
    
def GMM_output_transformation(output):
    # using notation from the book "Neural Networks for Pattern Recognition" (Bishop, 1995)
    M = 3 # number of gaussians
    c = 1 # dimensionality of target variables
    # extracting output activations
    z_alpha = output[:M,0]
    z_sigma = output[M:2*M,0]
    z_mu = output[2*M:,0]
    # converting output activations to parameters
    alphas = np.exp(z_alpha) / np.sum(np.exp(z_alpha))
    sigmas = np.exp(z_sigma)
    mus = z_mu

    return np.vstack((alphas, sigmas, mus))

def GMM_error_function(Y, Y_hat):
    # this implementation is horrible. Change asap.
    M = 3
    c = 1
        
    error = 0
    for data_idx in np.arange(Y_hat.shape[1]):
        for comp_idx in np.arange(M):
            y = Y[0,data_idx]
            curr_Y_hat = Y_hat[:,data_idx]
            alpha = curr_Y_hat[comp_idx]
            sigma = float(curr_Y_hat[comp_idx + M])
            print("sigma"  + str(sigma))
            mu    = curr_Y_hat[comp_idx + 2*M]
            prefactor = 1/((2*np.pi)**(c/2)*sigma**c)
            phi = prefactor * np.exp(-(y-mu)**2/(2*sigma**2))
            error += np.log( alpha * phi )
        
    return -error
                          
# create sample data
n_data = 100
c = 1
M = 3
x_dimension = 1
y_dimension = 1
n_loops = 10
hidden_layer_sizes = [30]
eta=.1
X = np.matrix(np.random.normal(size=(x_dimension, n_data)))
Y = np.matrix(np.empty((X.shape[1])))
for idx, x in enumerate(X.T):
    Y[0,idx] = x[0,0]**2 # + np.random.normal(scale=.5)

weights, errorvec = train(X, Y, hidden_layer_sizes=hidden_layer_sizes,
                          error_deriv=GMM_derivative,
                          n_outputs = (c+2)*M, n_loops=n_loops, eta=eta,
                          output_transformation = GMM_output_transformation,
                          error_function = GMM_error_function)

plt.figure()
plt.plot(np.arange(n_loops), errorvec)
plt.savefig('Error_over_iterations.png')

#Y_hat = classify(X, weights, GMM_output_transformation)
#error = GMM_error_function(Y, Y_hat)
#plt.figure()
#plt.plot(X[0,:].T, Y[0,:].T, "x", label="true")
#plt.plot(X[0,:].T, Y_hat[0,:].T, "x", label="predicted")
#plt.legend()
#plt.title("Error is: " + str(error))
#
#plt.figure()
#plt.scatter(Y[0,:], Y_hat[0,:])