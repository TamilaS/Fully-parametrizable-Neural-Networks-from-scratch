import numpy as np
from scipy.stats import truncnorm
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

#Sigmoid activation function for forward pass
@np.vectorize
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Sigmoid activation function for backward pass
@np.vectorize
def d_sigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

#ReLU activation function for forward pass
@np.vectorize
def relu (x):
    return np.maximum(x, 0)

#ReLU activation function for backward pass
@np.vectorize
def d_relu (x):
    return (np.sign(x) + 1) / 2
   
#loss method cross entropy
def cross_entropy(output, target):
    errors = -np.sum(target*np.log(output), axis=1)
    return errors/ len(errors)

# output function - softmax for forward pass
def softmax(x):
    a_exps = np.exp(x - x.max())
    return a_exps / np.sum(a_exps, axis=0)

#output activation function for backward pass - derivative of softmax
def d_softmax(x):
    a_exps = np.exp(x - x.max())
    return a_exps / np.sum(a_exps, axis=0) * (1 - a_exps / np.sum(a_exps, axis=0))

#for randomisation of creates parameters
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
        
    
class NeuralNetwork:
    
    def __init__(self, 
                 no_nodes,
                 learning_rate,
                 epochs):
        self.no_nodes = no_nodes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.create_weight_matrices()
        
    # bring as an output weigths list with weight considering input nodes and output nodes    
    def create_weight_matrices(self): 
        weights = []
        for i in range(len(self.no_nodes)-1):
            rad = 0.5 
            X = truncated_normal(mean=1, sd=1, low=-rad, upp=rad)
            weight = X.rvs((self.no_nodes[i], self.no_nodes[i+1]))
            weights.append(weight)
        return weights  
    
    #bias
    def f_bias (self):
        biases = []
        for i in range(1, len(self.no_nodes)):
            rad = 0.5 
            tn = truncated_normal(mean=2, sd=1, low=-rad, upp=rad)
            bias = tn.rvs(self.no_nodes[i]).reshape(-1,1) 
            biases.append(bias)
        return biases
    
    #forward pass
    def forward(self, X_train_trans):
        biases = self.f_bias()
        weights = self.create_weight_matrices()
        output_list = []
        for i in range(len(weights)):
            #input vector
            if i == 0:
                output_vector = np.dot(weights[i].T, X_train_trans) + biases[i]
                output_vector_in = activation_function(output_vector)
                output_list.append(output_vector_in)
            #output 
            elif i == (len(weights)-1):
                output_vector = np.dot (weights[i].T, output_list[i-1]) + biases[i]
                output_vector_out = softmax(output_vector) # softmax for the output layer
                output_list.append(output_vector_out)
            #hidden layers
            else:
                output_vector = np.dot (weights[i].T, output_list[i-1]) + biases[i]
                output_vector_out = activation_function(output_vector)
                output_list.append(output_vector_out)
        return output_vector_out, output_list
    

    #training with forward pass and backpropagation
    def train(self, X_train, y_train):
        weights = self.create_weight_matrices()
        # input_vector and target_vector can be tuple, list or ndarray
        X_train_trans = np.array(X_train, ndmin=2).T
        y_train_trans = np.array(y_train, ndmin=2).T
    
        for i in range(self.epochs):
        
            # forward pass
            forward = self.forward(X_train_trans)
            output = forward[0]            
            output_list = forward[1]
            # cost
            cost = cross_entropy(output, y_train_trans) # cross entropy
            #backprop   
            for i in reversed(range(len(weights))):
               
                if i == (len(weights)-1):
                    # derivative of the loss for the output
                    errors = (y_train_trans - output)
                    # derivative of the activation function
                    derivative = d_softmax (output)  #derivative of softmax for the output
                    tmp = errors * derivative
                    # multiply with the previous activation (output_vector_hidden)
                    who_update = self.learning_rate * (np.dot(tmp, output_list[i-1].T))
                    weights[i] += who_update.T 

                elif i == 0:
                    #from hidden to input layer
                    errors = np.dot(weights[i+1], errors * derivative)
                    derivative = activation_derivative(output_list[i])  
                    tmp = errors * derivative
                    wih_update = self.learning_rate * np.dot(tmp, X_train_trans.T)
                    weights[i] += wih_update.T

                elif i > 0 and i < (len(weights)-1):
                   # hidden layers
                    errors = np.dot(weights[i+1], errors * derivative)
                    derivative = activation_derivative(output_list[i])  
                    tmp = errors * derivative
                    whh_update = self.learning_rate * np.dot(tmp, output_list[i-1].T)
                    weights[i] += whh_update.T
                
        return weights, cost

    #testing predictions with new weights
    def run(self, X_test, weights):
        biases = self.f_bias()
        for i in range(len(weights)):
            #input layer
            if i == 0:
                input_vector = np.array(X_test, ndmin=2).T
                output_vector = np.dot(weights[0].T, input_vector) + biases[i]
                output_vector = activation_function(output_vector)
            #output layer    
            elif i == (len(weights)-1):
                output_vector = np.dot(weights[i].T, output_vector) + biases[i]
                output_vector = softmax(output_vector)    # softmax activation function for the output layer 
           #hidden layer
            else:
                output_vector = np.dot(weights[i].T, output_vector) + biases[i]
                output_vector = activation_function(output_vector)                
            y_hat = output_vector.T
        return y_hat
    
        #testing
    def test (self, y_hat, y_test): 

        y_hat = np.argmax(y_hat, axis=1)
        y_true = np.argmax(y_test, axis=1)
        correct = 0
        for pred, true in zip(y_hat, y_true):
            correct += 1 if pred == true else 0
        accuracy = correct / y_test.shape[0]
        print(f'Accuracy {accuracy}')
        
        

