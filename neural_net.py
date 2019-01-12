import numpy as np # linear algebra
import pandas as pd 
import os

def compute_num_layers(shape_list):
    return len(shape_list)-1

def sigmoid(X):
    return 1/(1+np.exp(-X))

def relu(X): 
    Z= np.copy(X)
    Z[X<0]=0
    return Z

def tanh(X):
    t=np.exp(2*X)
    return (t-1)/(t+1)

def binary_crossentropy(AL, Y):
    m= Y.shape[1]
    cost= np.squeeze(-np.sum((np.multiply(Y, np.log(AL))+ np.multiply((1-Y), np.log(1-AL))))/m)
    assert cost.shape==()
    return cost
    
def categorical_crossentropy(AL, Y):
    m= Y.shape[1]
    cost= - np.sum(np.multiply(Y, np.log(AL)))/m
    cost= np.squeeze(cost)
    assert cost.shape==()
    return cost

def compute_cost(AL, Y, loss="binary_crossentropy"):  
    
    assert AL.shape==Y.shape
    if loss== "categorical_crossentropy":
        cost= categorical_crossentropy(AL, Y)
        return cost
    elif loss=="binary_crossentropy":
        cost= binary_crossentropy(AL, Y)
        return cost
    else:
        print("Errrrrrrrrrrrr, give a proper loss metric")
        return None
    
def init_layer(shape_list):
    #This will return a cache(a dictionary) of Initialized weights per-layer and also the biases per layer
    # the shape list will have layers including the input layer dimensions. 
    weight_bias_cache={}
    for i in range(1, len(shape_list)):
        weight_bias_cache['W'+str(i)]= np.random.normal(0,np.sqrt(1/shape_list[i-1]), (shape_list[i-1],shape_list[i]))
        weight_bias_cache['b'+str(i)]= np.zeros((shape_list[i], 1))
    return weight_bias_cache

def forward_propagate_layer(A_prev, weight_bias_cache, layer_number,activation_preactivation_cache,activation='relu'):
    Wl= weight_bias_cache['W'+str(layer_number)]
    bl= weight_bias_cache['b'+str(layer_number)]
    Z= np.matmul(Wl.T, A_prev)+ bl
    if activation== 'relu':
        A = relu(Z)
    elif activation=='tanh':
        A= tanh(Z)
    else:
        A= sigmoid(Z)
    activation_preactivation_cache['A'+str(layer_number)]= A  
    activation_preactivation_cache['Z'+str(layer_number)]= Z
    return activation_preactivation_cache

def forward_propagation(shape_list, weight_bias_cache, X_input, activation_list):
    A_prev= X_input
    activation_preactivation_cache={}
    for i in range(1, len(shape_list)):
        activation_preactivation_cache=forward_propagate_layer(A_prev,weight_bias_cache,i,activation_preactivation_cache,activation=activation_list[i-1])  
        A_prev= activation_preactivation_cache["A"+str(i)]
    return activation_preactivation_cache

def dAL_binary_cross_entropy(Y, activation_preactivation_cache, shape_list, gradient_cache):
    num_layers= compute_num_layers(shape_list)
    AL= activation_preactivation_cache['A'+str(num_layers)]
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    gradient_cache['dA'+str(num_layers)]= dAL
    return gradient_cache

def sigmoid_backwards(Z,A, dA):
    #computes the gradient of the cost function with respect to the Z, if the activation of layer is sigmoid/''
    #note that,dA is the gradient wrt the same layer Activation 
    #whose gradient wrt Z we are evaluating
    assert A.shape==dA.shape
    assert Z.shape==A.shape
    dAdZ = np.multiply(A, (1-A))
    dZ= np.multiply(dA, dAdZ)
    assert dA.shape==dZ.shape
    return dZ

def relu_backwards(Z,A,dA):
    assert A.shape==dA.shape
    assert Z.shape==A.shape
    dZ= np.copy(Z)
    dZ[Z<0] =0
    dZ[Z>=0]=1
    dZ=np.multiply(dZ,dA) #chain rule
    return dZ

def activation_backwards(Z,A,dA,activation):
    if activation== 'sigmoid':
        return sigmoid_backwards(Z,A,dA)
    else:
        return relu_backwards(Z,A,dA)
    
def layer_backwards(dZ, layer_input_tuple):
    #layer input tuple contains the value of b, A_prev, W matrixx
    A_prev, W, b = layer_input_tuple
    m= A_prev.shape[1]
    dW= np.dot(dZ, A_prev.T).T/m
    db=np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev=  np.dot(W, dZ)
    return dA_prev, dW, db

def backpropagation(X_input, Y, activation_preactivation_cache, activation_list, shape_list, weight_bias_cache):
    gradient_cache={}
    gradient_cache= dAL_binary_cross_entropy(Y, activation_preactivation_cache, shape_list, gradient_cache)
    activation_preactivation_cache['A0']= X_input
    L= len(shape_list)
    for i in reversed(range(1,L)):
        Z= activation_preactivation_cache['Z'+str(i)]
        A= activation_preactivation_cache['A'+str(i)]
        A_prev=activation_preactivation_cache['A'+str(i-1)]
        W= weight_bias_cache['W'+str(i)]
        b= weight_bias_cache['b'+str(i)]
        dA= gradient_cache['dA'+str(i)]
        dZ= activation_backwards(Z,A,dA,activation_list[i-1])
        gradient_cache['dZ'+str(i)]= dZ
        dA_prev, dW, db= layer_backwards(dZ,(A_prev,W,b))
        gradient_cache['dW'+str(i)]=dW
        gradient_cache['db'+str(i)]=db
        gradient_cache['dA'+str(i-1)]=dA_prev
    return gradient_cache

def gradient_descent(weight_bias_cache, gradient_cache, shape_list,learning_rate=0.01):
    for i in range(1, len(shape_list)):
        weight_bias_cache['W'+str(i)]-= learning_rate*gradient_cache['dW'+str(i)]
        weight_bias_cache['b'+str(i)]-= learning_rate*gradient_cache['db'+str(i)]
    return weight_bias_cache

################ Partial Testing code################
X_input= np.array([1, 2, 3, 4, 5, 6,7,8,9]); X_input= np.reshape(X_input, (3,3))
Y_input=np.array([1, 0, 1]); Y_input=np.reshape(Y_input, (1,3))
shape_list=[3,2,1]
cost=[]
activation_list=['relu','sigmoid']
weight_bias_cache= init_layer(shape_list)

for i in range(0,10):
    activation_preactivation_cache= forward_propagation(shape_list,weight_bias_cache,X_input,activation_list)
    gradient_cache= backpropagation(X_input, Y_input,activation_preactivation_cache,activation_list,shape_list,weight_bias_cache)
    cost.append(compute_cost(activation_preactivation_cache['A'+str(len(shape_list)-1)], Y_input))
    gradient_descent(weight_bias_cache,gradient_cache,shape_list)


print(cost)