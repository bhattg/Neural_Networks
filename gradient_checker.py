import copy
class gradient_checker:
    def __init__(self, neural_network, epsilon):
        self.dummyNet= copy.deepcopy(neural_network)
        self.epsilon=epsilon
        
    def dict_to_vec():
        backprop_grads=[]
        #note that the order is dW then db
        for i in range(1,len(self.dummyNet.shape_list)):
            backprop_grads.append(self.dummyNet.gradient_cache['dW'+str(i)])
            backprop_grads.append(self.dummyNet.gradient_cache['db'+str(i)])
        self.backprop_grads=np.asarray(backprop_grads)
#     def evaluate_grad():
#         old= copy.deepcopy(self.dummyNet.weight_bias_cache)
#         for i in range(1,len(self.dummyNet.shape_list)):
#             self.dummyNet.weight_bias_cache['W'+str(i)]    
    
        