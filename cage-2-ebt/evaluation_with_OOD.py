import subprocess
import inspect
import time
from statistics import mean, stdev

import csv
import os
import sys

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.MainAgent import MainAgent
import random

import py_trees
import evaluation_bt_nodes_with_OOD as bt_nodes

import torch
from controller.ctrl import LSTMModel

import numpy as np

import random

import loaddata
import PNN_TEST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_EPS = 10
agent_name = 'Blue'
random.seed(0)

# Behavior tree

# Build blackboard
# Behavior tree

# Build blackboard
def build_blackboard():
    blackboard = py_trees.blackboard.Client(name = "Global")
    blackboard.register_key(key = "observation", access = py_trees.common.Access.WRITE)
    #Added for OODD 
    blackboard.register_key(key = "prev_observation", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "cur_observation", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "red_agent", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "OOD_Model", access = py_trees.common.Access.WRITE)

    blackboard.register_key(key = "action_space", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "transformed_action", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "cyborg", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "wrapped_cyborg", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)
    #blackboard.register_key(key = "reward", access = py_trees.common.Access.WRITE)

    blackboard.register_key(key = "start_actions", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "scan_state", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "agent_loaded", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "decoy_ids", access = py_trees.common.Access.WRITE)

    blackboard.register_key(key = "step", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "r", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "a", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "test_counter", access = py_trees.common.Access.WRITE)
    
    blackboard.register_key(key = "states", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "labels", access = py_trees.common.Access.WRITE)
    
    blackboard.register_key(key = "switch", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "window", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "lstm_model", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "need_switch", access = py_trees.common.Access.WRITE)

    #Add the two PNN Models
    blackboard.register_key(key = "PNN_Bline", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "PNN_Meander", access = py_trees.common.Access.WRITE)

    blackboard.register_key(key = "PNN_Bline_PrevState_Labels", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "PNN_Bline_CurState_Labels", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "PNN_Bline_Action_Labels", access = py_trees.common.Access.WRITE)

    blackboard.register_key(key = "PNN_Meander_PrevState_Labels", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "PNN_Meander_CurState_Labels", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "PNN_Meander_Action_Labels", access = py_trees.common.Access.WRITE)
    
    blackboard.register_key(key = "Distribution_Status", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "freq", access = py_trees.common.Access.WRITE)


    return blackboard


def build_bt(agent):
  
    #BT Root
    root = py_trees.composites.Sequence(name = "Cage Challenge BT", memory = True)

    #First Fallback
    determineStrat_sel = py_trees.composites.Selector(name = "Determine Strategy", memory = False)
    change_strat_check = bt_nodes.ChangeStratCheck()
    change_strat = bt_nodes.ChangeStrat()
    determineStrat_sel.add_children([change_strat_check, change_strat])

    #Second Fallback
    detectID_sel = py_trees.composites.Selector(name = "Detect ID", memory = False)
    detectID_check = bt_nodes.DetectIDCheck()
    getSafeAction = bt_nodes.GetSafeAction()
    detectID_sel.add_children([detectID_check, getSafeAction])
    
    #Third Fallback
    ODD_or_MainActionSeq = py_trees.composites.Selector(name = "Determine OOD or Execute Main Actions", memory = False)
    ODD_boolean = bt_nodes.Is_OOD()

    main_action_seq = py_trees.composites.Sequence(name = "Main Action Sequence", memory = True)

    get_ppo_action = bt_nodes.GetPPOAction()

    analyze_sel = py_trees.composites.Selector(name = "Analyze Selector", memory = False)
    analyze_check = bt_nodes.AnalyzeCheck()
    analyze = bt_nodes.Analyze()
    analyze_sel.add_children([analyze_check, analyze])
    analyze_check2 = bt_nodes.AnalyzeCheck()

    deploy_decoy_sel = py_trees.composites.Selector(name = "Deploy Decoy Selector", memory = False)
    deploy_decoy_check = bt_nodes.DeployDecoyCheck()
    deploy_decoy = bt_nodes.DeployDecoy()
    deploy_decoy_sel.add_children([deploy_decoy_check, deploy_decoy])
    deploy_decoy_check2 = bt_nodes.DeployDecoyCheck()

    # remove_decoys_sel = py_trees.composites.Selector(name = "Remove Decoys Selector", memory = False)
    # remove_decoys_check = bt_nodes.RemoveDecoysCheck()
    remove_decoys = bt_nodes.RemoveDecoys()
    # remove_decoys_sel.add_children([remove_decoys_check, remove_decoys])

    main_action_seq.add_children([get_ppo_action, analyze_sel, deploy_decoy_sel, analyze_check2, deploy_decoy_check2, remove_decoys])

    ODD_or_MainActionSeq.add_children([ODD_boolean,main_action_seq])
    
    root.add_children([determineStrat_sel,detectID_sel,ODD_or_MainActionSeq])
    #root.add_children([detectID_sel,ODD_or_MainActionSeq])
    
    return root


# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Blue')

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

class StratSwitch:
    def __init__(self, switch_step) -> None:
        self.switch_step = switch_step


if __name__ == "__main__":

    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    # commit_hash = get_git_revision_hash()
    commit_hash = "Not using git"
    # ask for a name
    name = "John Hannay"
    # ask for a team
    team = "CardiffUni"
    # ask for a name for the agent
    name_of_agent = "PPO + Greedy decoys"

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    save_file_name = "results/" + scenario + "_OODD"
    
    min_sw_step = 5
    max_sw_step = 100
    
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'
    switch = StratSwitch(switch_step=random.randint(min_sw_step, max_sw_step))

    agent = MainAgent()

    # Change this line to load your agentobservation
    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    print(f'using CybORG v{cyborg_version}, {scenario}\n')

    # Create LSTM Model
    INPUT_DIM = 52
    HIDDEN_DIM = 100
    LAYER_DIM = 2
    OUT_DIM = 1
    LEARNING_RATE = 1e-3

    lstm_model = LSTMModel(INPUT_DIM, HIDDEN_DIM, LAYER_DIM, OUT_DIM).to(device)
    MODEL_PATH = 'Models/controller/lstm_model.pth'
    lstm_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    lstm_model.eval()

    #Path to dataset
    path_train_bline = "/mnt/c/Users/Ankita/Downloads/cage-2-ebt/cage-2-ebt/Models/Dataset_train/bline/Steps_addn_100/"
    path_train_meander = "/mnt/c/Users/Ankita/Downloads/cage-2-ebt/cage-2-ebt/Models/Dataset_train/meander/Steps_addn_100/"

    #Generate Training data for agent against Bline
    S_bline,NS_bline,A_bline, R_bline = loaddata.load_ndarray(path_train_bline)
    S_bline_labels = loaddata.generate_labels(S_bline)
    S_bline_labelled = loaddata.label_NS(S_bline,S_bline_labels)
    
    NS_bline_labels = loaddata.generate_labels(NS_bline)
    NS_bline_labelled = loaddata.label_NS(NS_bline,NS_bline_labels) 
    
    Action_bline_labels = loaddata.generate_action_labels(A_bline)
    A_bline_labelled = loaddata.replace_action_with_labels(A_bline,Action_bline_labels)
    
    training_bline_data = np.concatenate((S_bline_labelled,A_bline_labelled,NS_bline_labelled),1)

    #Create a PNN for Bline
    Model_bline = PNN_TEST.Model((len(S_bline_labelled[0])+len(A_bline_labelled[0])),len(training_bline_data),len(NS_bline_labels),training_bline_data[:,:(len(S_bline_labelled[0])+len(A_bline_labelled[0]))],training_bline_data[:,(len(S_bline_labelled[0])+len(A_bline_labelled[0])):],len(A_bline_labelled[0]))
    print("Training Against Bline Agent In Progress")    
    Model_bline.train(training_bline_data[:,:(len(S_bline_labelled[0])+len(A_bline_labelled[0]))],training_bline_data[:,(len(S_bline_labelled[0])+len(A_bline_labelled[0])):],NS_bline_labelled) 

    #Generate Training data for agent against Meander
    S_meander,NS_meander,A_meander, R_meander = loaddata.load_ndarray(path_train_meander)
    S_meander_labels = loaddata.generate_labels(S_meander)
    S_meander_labelled = loaddata.label_NS(S_meander,S_meander_labels)
    
    NS_meander_labels = loaddata.generate_labels(NS_meander)
    NS_meander_labelled = loaddata.label_NS(NS_meander,NS_meander_labels) 
    
    Action_meander_labels = loaddata.generate_action_labels(A_meander)
    A_meander_labelled = loaddata.replace_action_with_labels(A_meander,Action_meander_labels)
        
    training_meander_data = np.concatenate((S_meander_labelled,A_meander_labelled,NS_meander_labelled),1)

    #Create a PNN for Meander
    Model_meander = PNN_TEST.Model((len(S_meander_labelled[0])+len(A_meander_labelled[0])),len(training_meander_data),len(NS_meander_labels),training_meander_data[:,:(len(S_meander_labelled[0])+len(A_meander_labelled[0]))],training_meander_data[:,(len(S_meander_labelled[0])+len(A_meander_labelled[0])):],len(A_meander_labelled[0]))
    print("Training Against Meander Agent In Progress")    
    Model_meander.train(training_meander_data[:,:(len(S_meander_labelled[0])+len(A_meander_labelled[0]))],training_meander_data[:,(len(S_meander_labelled[0])+len(A_meander_labelled[0])):],NS_meander_labelled) 

    rewards_list = []
    
    # Change this line to load your agentobservation
    for num_steps in [100]:
        # Create behavior tree 
        blackboard = build_blackboard()

        #Added for OODD
        blackboard.Distribution_Status = 1
        blackboard.PNN_Bline = Model_bline
        blackboard.PNN_Meander = Model_meander

        blackboard.PNN_Bline_PrevState_Labels = S_bline_labels
        blackboard.PNN_Bline_CurState_Labels = NS_bline_labels
        blackboard.PNN_Bline_Action_Labels = Action_bline_labels
        
        blackboard.PNN_Meander_PrevState_Labels = S_meander_labels
        blackboard.PNN_Meander_CurState_Labels = NS_meander_labels
        blackboard.PNN_Meander_Action_Labels = Action_meander_labels
        blackboard.freq = 0
                
        blackboard.switch = switch
        blackboard.lstm_model = lstm_model
        
        blackboard.states = []
        blackboard.labels = []

        blackboard.agent = agent
        red_agent = RedMeanderAgent
        #red_agent = B_lineAgent
        blackboard.red_agent = "RedMeanderAgent"
        #blackboard.red_agent = "B_lineAgent"
        blackboard.OOD_Model = blackboard.PNN_Meander
        #blackboard.OOD_Model = blackboard.PNN_Bline

        red2 = B_lineAgent
        #red2 = RedMeanderAgent
        blackboard.cyborg = CybORG(path, 'sim', agents={'Red': red_agent, 'Red2': red2}, strat_switch=switch)
        blackboard.wrapped_cyborg = wrap(blackboard.cyborg)

        blackboard.observation = blackboard.wrapped_cyborg.reset()

        #Added for OODD
        blackboard.prev_observation = blackboard.observation
        blackboard.action = -1

        blackboard.action_space = blackboard.wrapped_cyborg.get_action_space(agent_name)

        # action_space = cyborg.get_action_space(agent_name)
        total_reward = []
        actions = []

        with open(os.getcwd() + "/OOD_Results/Meander_" + str(num_steps) + "/Dist_Prob_wo_SelStrat.csv",'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["Episode", "Step_no", "Prev_State", "Prev_Action", "Current_State", "Freq", "Reward"])
        
        with open(os.getcwd() + "/OOD_Results/Meander_" + str(num_steps) + "/ID_wo_SelStrat.csv",'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["Episode", "Step_no", "Prev_State", "Prev_Action", "Current_State", "Reward"])
        
        with open(os.getcwd() + "/OOD_Results/Meander_" + str(num_steps) + "/OOD_wo_SelStrat.csv",'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["Episode", "Step_no", "Prev_State", "Prev_Action", "Current_State", "Reward"])
        
        with open(os.getcwd() + "/OOD_Results/Meander_" + str(num_steps) + "/OOD_wo_SelStratStep.csv",'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["Episode", "Step_no"])
        
        

        #with open(os.getcwd() + "/OOD_Results/Meander_" + str(num_steps) + "/ID_addn_1_RestoreRev.csv",'w', newline='') as outcsv:
        #    writer = csv.writer(outcsv)
        #    writer.writerow(["Episode", "Step_no", "Prev_State", "Prev_Action", "Current_State", "Reward"])
        
        #with open(os.getcwd() + "/OOD_Results/Meander_" + str(num_steps) + "/OOD_addn_1_RestoreRev.csv",'w', newline='') as outcsv:
        #    writer = csv.writer(outcsv)
        #    writer.writerow(["Episode", "Step_no", "Prev_State", "Prev_Action", "Current_State", "Reward"])


        for i in range(MAX_EPS):
            print("EPISODE",i)
            blackboard.r = []
            blackboard.a = []
            blackboard.window = []
            blackboard.need_switch = True

            root = build_bt(agent)

            blackboard.test_counter = 0
            blackboard.step = 0

            blackboard.agent = agent

            blackboard.decoy_ids = list(range(1000, 1009))
            blackboard.action_space = [133, 134, 135, 139, 3, 4, 5, 9, 16, 17, 18, 22, 11, 12, 13, 14,
                                            141, 142, 143, 144, 132, 2, 15, 24, 25, 26, 27] + blackboard.decoy_ids
            
            blackboard.scan_state = np.zeros(10)
            blackboard.start_actions = [51, 116, 55]

            blackboard.Distribution_Status = 1
            reward = 0.0
            blackboard.switch.switch_step = random.randint(min_sw_step, max_sw_step)
            # subtract 3 because of setup steps
            for j in range(num_steps):
                if blackboard.step < 3:
                    blackboard.cur_observation = blackboard.observation
                    blackboard.agent.add_scan(blackboard.observation)
                    if len(blackboard.agent.start_actions) > 0:
                        blackboard.transformed_action = blackboard.agent.start_actions[0]
                        blackboard.agent.start_actions = blackboard.agent.start_actions[1:]
                else:
                    print(blackboard.prev_observation)
                    print(blackboard.cur_observation)  
                    print(blackboard.transformed_action) 
                    
                    root.tick_once()
                
            

                #if blackboard.Distribution_Status == 0:
                #    with open(os.getcwd() + "/OOD_Results/Meander_" + str(num_steps) + "/OOD_WO_Monitor.csv",'a', newline='') as outcsv:
                #        writer = csv.writer(outcsv)
                #        writer.writerow([str(i), str(blackboard.step),str(blackboard.prev_observation), str(blackboard.transformed_action), str(blackboard.cur_observation), str(reward)])
                #else:
                #    with open(os.getcwd() + "/OOD_Results/Meander_" + str(num_steps) + "/ID_WO_Monitor.csv",'a', newline='') as outcsv:
                #        writer = csv.writer(outcsv)
                #        writer.writerow([str(i), str(blackboard.step),str(blackboard.prev_observation), str(blackboard.transformed_action), str(blackboard.cur_observation), str(reward)])

                blackboard.prev_observation = blackboard.cur_observation

                blackboard.observation, reward, done, info = blackboard.wrapped_cyborg.step(blackboard.transformed_action)
                
                blackboard.cur_observation = blackboard.observation
                blackboard.r.append(reward)

                blackboard.states.append(blackboard.observation)
                blackboard.window.append(blackboard.observation)
                if len(blackboard.window) > 5:
                    blackboard.window.pop(0)
                if blackboard.step < blackboard.switch.switch_step:
                    blackboard.labels.append([0])
                else:
                    blackboard.labels.append([1])
                blackboard.a.append((str(blackboard.cyborg.get_last_action('Blue')),
                                        str(blackboard.cyborg.get_last_action('Red'))))

                blackboard.step += 1

                if blackboard.step > 3:
                    with open(os.getcwd() + "/OOD_Results/Meander_" + str(num_steps) + "/Dist_Prob_wo_SelStrat.csv",'a', newline='') as outcsv:
                        writer = csv.writer(outcsv)
                        writer.writerow([str(i), str(blackboard.step),str(blackboard.prev_observation), str(blackboard.transformed_action), str(blackboard.cur_observation), str(blackboard.freq), str(reward)])

                if blackboard.Distribution_Status == 0:
                    with open(os.getcwd() + "/OOD_Results/Meander_" + str(num_steps) + "/OOD_wo_SelStrat.csv",'a', newline='') as outcsv:
                        writer = csv.writer(outcsv)
                        writer.writerow([str(i), str(blackboard.step),str(blackboard.prev_observation), str(blackboard.transformed_action), str(blackboard.cur_observation), str(reward)])
                else:
                    with open(os.getcwd() + "/OOD_Results/Meander_" + str(num_steps) + "/ID_wo_SelStrat.csv",'a', newline='') as outcsv:
                        writer = csv.writer(outcsv)
                        writer.writerow([str(i), str(blackboard.step),str(blackboard.prev_observation), str(blackboard.transformed_action), str(blackboard.cur_observation), str(reward)])

            agent.end_episode()

            with open(os.getcwd() + "/OOD_Results/Meander_" + str(num_steps) + "/OOD_wo_SelStratStep.csv",'a', newline='') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerow([str(i), str(switch.switch_step)])

            rewards_list.append(blackboard.r)
            total_reward.append(sum(blackboard.r))
            actions.append(blackboard.a)
            blackboard.observation = blackboard.wrapped_cyborg.reset()
            print("ep done. reward is: ", sum(blackboard.r))
            blackboard.switch.switch_step = random.randint(min_sw_step, max_sw_step)

    np.save(save_file_name + ".npy", np.array(rewards_list))
    