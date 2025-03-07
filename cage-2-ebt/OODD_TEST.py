import numpy as np
import torch
import sys
import time
import random 
import math
import csv


import loaddata #type: ignore
import PNN_TEST #type: ignore

if __name__ == '__main__':

    path_train = "/mnt/c/Users/Ankita/Downloads/cage-2-ebt/cage-2-ebt/Models/Dataset_train/bline/Steps_addn_100/"
    #path_test = "/mnt/c/Users/Ankita/Downloads/cage-2-ebt/cage-2-ebt/Models/Dataset_train/bline/Steps_100/"
    path_test = "/mnt/c/Users/Ankita/Downloads/cage-2-ebt/cage-2-ebt/emu_data/"

    with open('./emu_data/bline_emu_ood_episodes.csv', 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["Iteration", "ID", "ONT", "TNT", "ANF", "SNF"])

    #f = open("./Results/bline_sim_vs_emu.csv","w")

    with open('./emu_data/bline_emu_rewards.csv', 'w', newline='') as outcsv:
        writer1 = csv.writer(outcsv)
        writer1.writerow(["Iteration", "IDR", "OODR"])


    S,NS,A,R_train = loaddata.load_ndarray(path_train)
    S_labels = loaddata.generate_labels(S)
    S_labelled = loaddata.label_NS(S,S_labels)
    NS_labels = loaddata.generate_labels(NS)
    NS_labelled = loaddata.label_NS(NS,NS_labels) 
    Action_labels = loaddata.generate_action_labels(A)
    A_labelled = loaddata.replace_action_with_labels(A,Action_labels)
        
    training_data = np.concatenate((S_labelled,A_labelled,NS_labelled),1)

    Rewards_ID = []
    Rewards_OOD = []
    #print(len(S_labels))
    #print(len(NS_labels))
    #print(len(Action_labels))
    Model = PNN_TEST.Model((len(S_labelled[0])+len(A_labelled[0])),len(training_data),len(NS_labels),training_data[:,:(len(S_labelled[0])+len(A_labelled[0]))],training_data[:,(len(S_labelled[0])+len(A_labelled[0])):],len(A_labelled[0]))
    print("Training In Progress")    
    Model.train(training_data[:,:(len(S_labelled[0])+len(A_labelled[0]))],training_data[:,(len(S_labelled[0])+len(A_labelled[0])):],NS_labelled)    
    
    print("Testing in Progress")
    S_test,NS_test,A_test,R_test = loaddata.load_ndarray(path_test)        
    test_data = np.concatenate((S_test,A_test,NS_test),1)
    #print(R_test)

    for i in range(len(test_data)):
        print("--------------------------------Test Traces-----------------------------------------")
        print("Test Instance " + str(i+1))
        S_test_labelled = loaddata.get_label_NS(S_test[i],S_labels)
        NS_test_labelled = loaddata.get_label_NS(NS_test[i],NS_labels) 
        A_test_labelled = loaddata.get_label_action(A_test[i],Action_labels)
        if S_test_labelled > 0 and NS_test_labelled > 0 and A_test_labelled > 0:
            test_data_sample = []
            test_data_sample.append(S_test_labelled)
            test_data_sample.append(A_test_labelled)
            test_data_sample.append(NS_test_labelled)
            status = Model.test(test_data_sample[0:2],test_data_sample[2:])
            if status == -1:
                #Transition to output state not in training data
                print("No Transition to Next State")
                Rewards_OOD.append(R_test[i])
                with open('./emu_data/bline_emu_ood_episodes.csv', 'a', newline='') as outcsv:
                   writer = csv.writer(outcsv)
                   writer.writerow([i+1, None, None, 3, None, None])
                with open('./Results/bline_emu_rewards.csv', 'a', newline='') as outcsv:
                   writer1 = csv.writer(outcsv)
                   writer1.writerow([i+1, None, R_test[i]])

            elif status == -2:
                #Misprediction to different output state
                print("Transition to output state not in training data")
                Rewards_OOD.append(R_test[i])
                with open('./emu_data/bline_emu_ood_episodes.csv', 'a', newline='') as outcsv:
                    writer = csv.writer(outcsv)
                    writer.writerow([i+1, None, 2, None, None, None])
                with open('./emu_data/bline_emu_rewards.csv', 'a', newline='') as outcsv:
                    writer1 = csv.writer(outcsv)
                    writer1.writerow([i+1, None, R_test[i]])
            else:
                print("In Distribution")
                Rewards_ID.append(R_test[i])
                with open('./emu_data/bline_emu_rewards.csv', 'a', newline='') as outcsv:
                    writer1 = csv.writer(outcsv)
                    writer1.writerow([i+1, R_test[i], None])
                with open('./emu_data/bline_emu_ood_episodes.csv', 'a', newline='') as outcsv:
                    writer = csv.writer(outcsv)
                    writer.writerow([i+1, 1, None, None, None, None])
                
        else:
            if S_test_labelled == 0:
                #Current state not in dataset
                print("Current State is not in training data")
                Rewards_OOD.append(R_test[i])
                with open('./emu_data/bline_emu_ood_episodes.csv', 'a', newline='') as outcsv:
                    writer = csv.writer(outcsv)
                    writer.writerow([i+1, None, None, None, None, 5])
                with open('./emu_data/bline_emu_rewards.csv', 'a', newline='') as outcsv:
                    writer1 = csv.writer(outcsv)
                    writer1.writerow([i+1, None, R_test[i]])
            #    f.write(str(i+1) + " " + "5 " + "\n")
            #if NS_test_labelled == 0:
                #Next state not in dataset
            #    print("Next State is not in training data")
            #    Rewards_OOD.append(R[i])
                #with open('./Results/bline_sim_30_1.csv', 'a', newline='') as outcsv:
                #    writer = csv.writer(outcsv)
                #    writer.writerow([i+1, None, None, None, None, 6, None])
            else:
                #Action not in dataset
                print("Action is not in training data")
                Rewards_OOD.append(R_test[i])
                with open('./emu_data/bline_emu_ood_episodes.csv', 'a', newline='') as outcsv:
                    writer = csv.writer(outcsv)
                    writer.writerow([i+1, None, None, None, 4, None])
                with open('./emu_data/bline_emu_rewards.csv', 'a', newline='') as outcsv:
                    writer1 = csv.writer(outcsv)
                    writer1.writerow([i+1, None, R_test[i]])
        print("------------------------------------------------------------------------------------")

    