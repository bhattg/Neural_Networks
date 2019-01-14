import numpy as np # linear algebra
import pandas as pd 
import os
import copy

def make_batches(X_train, Y_train, batch_size=32):
    m= X_train.shape[0]
    num_batches=0
    batch_list=[]
    if m%batch_size==0:
        num_batches=(int)(m/batch_size)
    else:
        num_batches=int(int(m)/int(batch_size))+1
    for i in range(0,int(num_batches)):
        start_index= i*batch_size
        if i==num_batches-1:
            end_index=m
        else :
            end_index=(i+1)*batch_size
        X_temp= X_train[start_index:end_index, :]
        Y_temp= Y_train[start_index:end_index, :]
        batch_list.append([X_temp, Y_temp])
    np.random.shuffle(batch_list)
    return batch_list

class Neural_Network:
    
    def __compute_num_layers(self):
        return len(self.shape_list)-1

    def __sigmoid(self,X):
        return 1/(1+np.exp(-X))

    def __relu(self,X): 
        Z= np.copy(X)
        Z[X<0]=0
        return Z

    def __tanh(self,X):
        t=np.exp(2*X)
        return (t-1)/(t+1)

    def __binary_crossentropy(self,AL, Y):
        m= Y.shape[0]
        cost= np.squeeze(-np.sum((np.multiply(Y, np.log(AL))+ np.multiply((1-Y), np.log(1-AL))))/m);assert cost.shape==()
        return cost
    
    def __categorical_crossentropy(self,AL, Y):
        m= Y.shape[0]
        cost= - np.sum(np.multiply(Y, np.log(AL)))/m
        cost= np.squeeze(cost)
        assert cost.shape==()
        return cost

    def __compute_cost(self,loss="binary_crossentropy"):  
        last_layer_index= self.__compute_num_layers()
        AL= self.activation_preactivation_cache['A'+str(last_layer_index)]
        Y= self.Y_train
        assert AL.shape==Y.shape
        if loss== "categorical_crossentropy":
            cost= self.__categorical_crossentropy(AL, Y)
            return cost
        elif loss=="binary_crossentropy":
            cost= self.__binary_crossentropy(AL, Y)
            return cost
        else:
            print("Errrrrrrrrrrrr, give a proper loss metric")
            return None
        
    def compute_cost(self,loss="binary_crossentropy"):  
        last_layer_index= self.__compute_num_layers()
        AL= self.activation_preactivation_cache['A'+str(last_layer_index)]
        Y= self.Y_train
        assert AL.shape==Y.shape
        if loss== "categorical_crossentropy":
            cost= self.__categorical_crossentropy(AL, Y)
            return cost
        elif loss=="binary_crossentropy":
            cost= self.__binary_crossentropy(AL, Y)
            return cost
        else:
            print("Errrrrrrrrrrrr, give a proper loss metric")
            return None

    def __init_layer(self):
        #This will return a cache(a dictionary) of Initialized weights per-layer and also the biases per layer
        # the shape list will have layers including the input layer dimensions. 
        for i in range(1, len(self.shape_list)):
            self.weight_bias_cache['W'+str(i)]= np.random.normal(0,np.sqrt(1/self.shape_list[i-1]), (self.shape_list[i-1],self.shape_list[i]))
            self.weight_bias_cache['b'+str(i)]= np.zeros((1,self.shape_list[i]))
    
    def __forward_propagate_layer(self,A_prev, layer_number,activation='relu'):
        Wl= self.weight_bias_cache['W'+str(layer_number)]
        bl= self.weight_bias_cache['b'+str(layer_number)]
        Z= np.dot(A_prev, Wl)+ bl
        if activation== 'relu':
            A = self.__relu(Z)
        elif activation=='tanh':
            A= self.__tanh(Z)
        else:
            A= self.__sigmoid(Z)
        self.activation_preactivation_cache['A'+str(layer_number)]= A  
        self.activation_preactivation_cache['Z'+str(layer_number)]= Z
        

    def __forward_propagation(self):
        A_prev= self.X_train
        for i in range(1, len(self.shape_list)):
            self.__forward_propagate_layer(A_prev,i,activation=self.activation_list[i-1])  
            A_prev= self.activation_preactivation_cache["A"+str(i)]
    
    def forward_propagation(self):
        A_prev= self.X_train
        for i in range(1, len(self.shape_list)):
            self.__forward_propagate_layer(A_prev,i,activation=self.activation_list[i-1])  
            A_prev= self.activation_preactivation_cache["A"+str(i)]
    
    
    def __dAL_binary_cross_entropy(self,Y):
        num_layers= self.__compute_num_layers()
        AL= self.activation_preactivation_cache['A'+str(num_layers)]
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        self.gradient_cache['dA'+str(num_layers)]= dAL

    def __sigmoid_backwards(self,Z,A, dA): 
        #computes the gradient of the cost function with respect to the Z, if the activation of layer is sigmoid/''
        #note that,dA is the gradient wrt the same layer Activation 
        #whose gradient wrt Z we are evaluating
        assert A.shape==dA.shape
        assert Z.shape==A.shape
        dAdZ = np.multiply(A, (1-A))
        dZ= np.multiply(dA, dAdZ)
        assert dA.shape==dZ.shape
        return dZ

    def __relu_backwards(self,Z,A,dA):
        assert A.shape==dA.shape
        assert Z.shape==A.shape
        dZ= np.copy(Z)
        dZ[Z<0] =0
        dZ[Z>=0]=1
        dZ=np.multiply(dZ,dA) #chain rule
        return dZ

    def __activation_backwards(self,Z,A,dA,activation):
        if activation== 'sigmoid':
            return self.__sigmoid_backwards(Z,A,dA)
        else:
            return self.__relu_backwards(Z,A,dA)

    def __layer_backwards(self,dZ, layer_input_tuple):
        #layer input tuple contains the value of b, A_prev, W matrixx
        A_prev, W, b = layer_input_tuple
        m= A_prev.shape[0]
        dW= np.dot(A_prev.T,dZ)/m
        db=np.sum(dZ, axis=0, keepdims=True)/m
        dA_prev=  np.dot(dZ, W.T)
        return dA_prev, dW, db

    def __backpropagation(self):
        self.__dAL_binary_cross_entropy(self.Y_train)
        self.activation_preactivation_cache['A0']= self.X_train
        L= len(self.shape_list)
        for i in reversed(range(1,L)):
            Z= self.activation_preactivation_cache['Z'+str(i)]
            A= self.activation_preactivation_cache['A'+str(i)]
            A_prev=self.activation_preactivation_cache['A'+str(i-1)]
            W= self.weight_bias_cache['W'+str(i)]
            b= self.weight_bias_cache['b'+str(i)]
            dA= self.gradient_cache['dA'+str(i)]
            dZ= self.__activation_backwards(Z,A,dA,self.activation_list[i-1])
            self.gradient_cache['dZ'+str(i)]= dZ
            dA_prev, dW, db= self.__layer_backwards(dZ,(A_prev,W,b))
            self.gradient_cache['dW'+str(i)]=dW
            self.gradient_cache['db'+str(i)]=db
            self.gradient_cache['dA'+str(i-1)]=dA_prev
        
    def __update_parameters(self):
        for i in range(1, len(self.shape_list)):
            
            self.weight_bias_cache['W'+str(i)]-= self.learning_rate*self.gradient_cache['dW'+str(i)]
            self.weight_bias_cache['b'+str(i)]-= self.learning_rate*self.gradient_cache['db'+str(i)]

    def __init__(self, shape_list,activation_list,epochs=20,loss='binary_crossentropy',learning_rate=0.001,verbose=False):
        self.cost=[]
        self.verbose=verbose
        self.activation_preactivation_cache={}
        self.gradient_cache={}
        self.shape_list=shape_list
        self.weight_bias_cache={}
        self.activation_list=activation_list
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.loss=loss
        self.__init_layer()
        
    def fit(self,X_train,Y_train):
        self.X_train=X_train
        self.Y_train=Y_train
        for i in range(0, self.epochs):
            self.__forward_propagation()
            self.cost.append(self.__compute_cost())
            if self.verbose:
                print(self.cost[-1])
            self.__backpropagation()
            self.__update_parameters()
            
    def __forward_propagate_prediction(self):
        A_prev= self.X_test
        for i in range(1, len(self.shape_list)):
            self.__forward_propagate_layer(A_prev,i,activation=self.activation_list[i-1])  
            A_prev= self.activation_preactivation_cache["A"+str(i)]
        AL= self.activation_preactivation_cache['A'+str(len(self.shape_list)-1)]
        return AL
    
    def predict(self, X_test):
        self.X_test=X_test
        AL= self.__forward_propagate_prediction()
        Y_hat= np.copy(AL)
        if self.activation_list[-1]=='sigmoid':
            Y_hat[AL<0.5]=0
            Y_hat[AL>0.5]=1
            return Y_hat
        else:
            print('softmax yet to be implemented')
            return 
        
        
    def score(self,X_test, Y_test):
        self.X_test=X_test
        AL= self.__forward_propagate_prediction()
        Y_hat= np.copy(AL)
        if self.activation_list[-1]=='sigmoid':
            Y_hat[AL<0.5]=0
            Y_hat[AL>0.5]=1
        assert Y_hat.shape==Y_test.shape
        Y=np.equal(Y_hat, Y_test)
        return np.sum(Y)/Y_test.shape[0]
        

################ Partial Testing code################
X_all, Y_all= load_breast_cancer(return_X_y=True)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X_all, Y_all, test_size=0.2, random_state=22)
Y_train= np.reshape(Y_train, (Y_train.shape[0], 1))
Y_test= np.reshape(Y_test, (Y_test.shape[0], 1))
from sklearn.preprocessing import StandardScaler
s= StandardScaler()
s.fit(X_train)
X_train= s.transform(X_train)
X_test= s.transform(X_test)

########from our implementation

mini=110.0; maxa=0.0
for i in range(0, 100):
    nn= Neural_Network([30,20,10,5, 1], ['relu','relu','relu', 'sigmoid'],epochs=50)
    nn.fit(X_train,Y_train)
    acc=nn.score(X_test, Y_test)*100
    if acc<mini:
        mini=acc
    elif acc>maxa:
        maxa=acc
print('max ', maxa)
print('min ', mini)

############## inbuild

from sklearn.neural_network import MLPClassifier

clf= MLPClassifier(hidden_layer_sizes=(30,20,10,5,))
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)