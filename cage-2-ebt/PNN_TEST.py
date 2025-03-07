import numpy as np
import torch
import math

class Model():
    def __init__(self, input_dim, pattern_dim, summation_dim, input, output, action_dim):
        
        self.input_dim = input_dim
        self.pattern_dim = pattern_dim
        self.summation_dim = summation_dim
        self.action_dim = action_dim
        self.input = input
        self.output = output

        #Input Vector
        shape = (self.input_dim)
        self.layer_1 = np.zeros(shape)

        #Pattern Layer 
        shape = (self.pattern_dim,self.input_dim)
        self.layer_2 = np.zeros(shape)

        self.connection_1 = {}

        #Summation_Layer output classification
        self.connection_2 = []
        for i in range(self.summation_dim):
            self.connection_2.append([])
        
        #Output layer
        self.pattern_out = np.zeros(self.summation_dim)
        
    def train(self, input , output, NS_labelled):
        #Generate Layer 2
        for i in range(self.pattern_dim):
            self.layer_2[i,:] = input[i,:self.input_dim]

        #Generate Connection 1 and 2
        for i in range(self.pattern_dim):
            self.connection_1[i+1] = output[i][0]
            self.connection_2[(output[i][0]-1)].append(i)
        #print(self.connection_2)
    
    def test(self, input, output):
        self.layer_1[:] = input[:]   
        
        list_of_inp = []
        for i in range(self.pattern_dim):
            if (self.layer_1[:] == self.layer_2[i,:]).all():
                list_of_inp.append(i+1)
        #print(list_of_inp)
        
        list_of_op = {}
        for i in range(len(list_of_inp)):
            if self.connection_1[list_of_inp[i]] not in list_of_op.keys():
                list_of_op[self.connection_1[list_of_inp[i]]] = 1
            else:
                list_of_op[self.connection_1[list_of_inp[i]]] += 1
        #print(list_of_op)

        if len(list_of_op) == 0:
            print("No Transition from current state is in Distribution")
            return -1,0
        else:
            Prob_out = []
            found = False
            #print(list_of_op.keys)
            for keys in list_of_op.keys():
                denom = len(list_of_inp)
                num = list_of_op[keys]
                prob = num/denom
                Prob_out.append(prob)
                list_of_op[keys] = prob
                print("Predicted Probability of output state " + str(keys) + " : " + str(prob))
                if output[0] == keys:
                    found = True

            if found == False:
                print("Transition to Output State is not in Training Data")
                return -2,0
            else:
                print("Actual Output State " + str(output[0]))
                return 0,num