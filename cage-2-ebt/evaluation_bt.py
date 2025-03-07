import subprocess
import inspect
import time
from statistics import mean, stdev
import numpy as np

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.MainAgent import MainAgent
import random

import py_trees
import evaluation_bt_nodes as bt_nodes

from metrics import precision, recall , precision_test
import csv

#from vu_emu import vu_emu
import json
from utils import *
import ast
import argparse
from copy import deepcopy
from statistics import mean, stdev
from typing import Optional
from datetime import datetime
import copy

import loaddata
import PNN_TEST

MAX_EPS = 1 
agent_name = 'Blue'
random.seed(0)

# Behavior tree

# Build blackboard
def build_blackboard():
    blackboard = py_trees.blackboard.Client(name = "Global")
    blackboard.register_key(key = "observation", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "action_space", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "cyborg", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "wrapped_cyborg", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)
    #blackboard.register_key(key = "reward", access = py_trees.common.Access.WRITE)

    blackboard.register_key(key = "start_actions", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "scan_state", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "agent_loaded", access = py_trees.common.Access.WRITE)

    blackboard.register_key(key = "step", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "r", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "a", access = py_trees.common.Access.WRITE)
    blackboard.register_key(key = "test_counter", access = py_trees.common.Access.WRITE)

    return blackboard

def build_bt(agent):
    #New Behavior Tree with OODD detection
    #root = py_trees.composites.Selector(name = "CAGE Challenge BT", memory = False)

    #Initial_steps = py_trees.composites.Sequence(name = "Strat Select and OODD Detect", memory = True)
    
    root = py_trees.composites.Sequence(name = "CAGE Challenge BT", memory = True)

    determine_action = py_trees.composites.Selector(name = "Determine Action", memory = False)
    
    setup_seq = py_trees.composites.Sequence(name = "Setup Steps", memory = True)
    setup_check = bt_nodes.SetupCheck()
    setup = bt_nodes.Setup(agent)
    setup_seq.add_children([setup_check, setup])

    main_action_seq = py_trees.composites.Sequence(name = "Main Action Steps", memory = True)

    change_strat_sel = py_trees.composites.Selector(name = "Change Strategy Selector", memory = False)
    change_strat_check = bt_nodes.ChangeStratCheck()
    change_strat_check_inv = py_trees.decorators.Inverter(name = "Inverter", child = change_strat_check)
    change_strat = bt_nodes.ChangeStrat()
    change_strat_sel.add_children([change_strat_check_inv, change_strat])

    get_ppo_action = bt_nodes.GetPPOAction()

    deploy_decoy_sel = py_trees.composites.Selector(name = "Deploy Decoy Selector", memory = False)
    deploy_decoy_check = bt_nodes.DeployDecoyCheck()
    deploy_decoy_check_inv = py_trees.decorators.Inverter(name = "Inverter", child = deploy_decoy_check)
    deploy_decoy = bt_nodes.DeployDecoy()
    deploy_decoy_sel.add_children([deploy_decoy_check_inv, deploy_decoy])

    remove_decoys_sel = py_trees.composites.Selector(name = "Remove Decoys Selector", memory = False)
    remove_decoys_check = bt_nodes.RemoveDecoysCheck()
    remove_decoys_check_inv = py_trees.decorators.Inverter(name = "Inverter", child = remove_decoys_check)
    remove_decoys = bt_nodes.RemoveDecoys()
    remove_decoys_sel.add_children([remove_decoys_check_inv, remove_decoys])

    main_action_seq.add_children([change_strat_sel, get_ppo_action, deploy_decoy_sel, remove_decoys_sel])

    determine_action.add_children([setup_seq, main_action_seq])

    execute_actions = bt_nodes.ExecuteActions()

    root.add_children([determine_action, execute_actions])

    # setup = bt_nodes.Setup(agent)
    # get_action = bt_nodes.GetAction()
    # take_action = bt_nodes.TakeAction()
    # root.add_children([setup, get_action, take_action])

    #py_trees.display.render_dot_tree(root)
    return root



# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Blue')

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

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

    agent = MainAgent()

    # Change this line to load your agentobservation

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    #file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    #print(f'Saving evaluation results to {file_name}')
    #with open(file_name, 'a+') as data:
    #    data.write(f'CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n')
    #    data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
    #    data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    print(f'using CybORG v{cyborg_version}, {scenario}\n')

    #Path to dataset
    #path_train_bline = "/mnt/c/Users/Ankita/Downloads/cage-2-ebt/cage-2-ebt/Models/Dataset_train/bline/Steps_200/"
    #path_train_meander = "/mnt/c/Users/Ankita/Downloads/cage-2-ebt/cage-2-ebt/Models/Dataset_train/meander/Steps_200/"

    #Generate Training data for agent against Bline
    #S_bline,NS_bline,A_bline = loaddata.load_ndarray(path_train_bline)
    #S_bline_labels = loaddata.generate_labels(S_bline)
    #S_bline_labelled = loaddata.label_NS(S_bline,S_bline_labels)
    #NS_bline_labels = loaddata.generate_labels(NS_bline)
    #NS_bline_labelled = loaddata.label_NS(NS_bline,NS_bline_labels) 
    #Action_bline_labels = loaddata.generate_action_labels(A_bline)
    #A_bline_labelled = loaddata.replace_action_with_labels(A_bline,Action_bline_labels)
    
    #training_bline_data = np.concatenate((S_bline_labelled,A_bline_labelled,NS_bline_labelled),1)

    #Create a PNN for Bline
    #Model_bline = PNN_TEST.Model((len(S_bline_labelled[0])+len(A_bline_labelled[0])),len(training_bline_data),len(NS_bline_labels),training_bline_data[:,:(len(S_bline_labelled[0])+len(A_bline_labelled[0]))],training_bline_data[:,(len(S_bline_labelled[0])+len(A_bline_labelled[0])):],len(A_bline_labelled[0]))
    #print("Training Against Bline Agent In Progress")    
    #Model_bline.train(training_bline_data[:,:(len(S_bline_labelled[0])+len(A_bline_labelled[0]))],training_bline_data[:,(len(S_bline_labelled[0])+len(A_bline_labelled[0])):],NS_bline_labelled) 

    #Generate Training data for agent against Meander
    #S_meander,NS_meander,A_meander = loaddata.load_ndarray(path_train_meander)
    #S_meander_labels = loaddata.generate_labels(S_meander)
    #S_meander_labelled = loaddata.label_NS(S_meander,S_meander_labels)
    #NS_meander_labels = loaddata.generate_labels(NS_meander)
    #NS_meander_labelled = loaddata.label_NS(NS_meander,NS_meander_labels) 
    #Action_meander_labels = loaddata.generate_action_labels(A_meander)
    #A_meander_labelled = loaddata.replace_action_with_labels(A_meander,Action_meander_labels)
        
    #training_meander_data = np.concatenate((S_meander_labelled,A_meander_labelled,NS_meander_labelled),1)

    #Create a PNN for Meander
    #Model_meander = PNN_TEST.Model((len(S_meander_labelled[0])+len(A_meander_labelled[0])),len(training_meander_data),len(NS_meander_labels),training_meander_data[:,:(len(S_meander_labelled[0])+len(A_meander_labelled[0]))],training_meander_data[:,(len(S_meander_labelled[0])+len(A_meander_labelled[0])):],len(A_meander_labelled[0]))
    #print("Training Against Meander Agent In Progress")    
    #Model_meander.train(training_meander_data[:,:(len(S_meander_labelled[0])+len(A_meander_labelled[0]))],training_meander_data[:,(len(S_meander_labelled[0])+len(A_meander_labelled[0])):],NS_meander_labelled) 
    
    for num_steps in [50]:
        for red_agent in [B_lineAgent]:

            # Create behavior tree 
            blackboard = build_blackboard()

            blackboard.agent = agent
            
            blackboard.cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            blackboard.wrapped_cyborg = wrap(blackboard.cyborg)

            blackboard.observation = blackboard.wrapped_cyborg.reset()
            # observation = cyborg.reset().observation

            blackboard.action_space = blackboard.wrapped_cyborg.get_action_space(agent_name)

            
            # action_space = cyborg.get_action_space(agent_name)
            total_reward = []
            actions = []
            if red_agent != SleepAgent:
                for i in range(MAX_EPS):
                    # print(i)
                    blackboard.r = []
                    blackboard.a = []

                    root = build_bt(agent)
                    

                    # get_action.setup()  # initialize the parameters for episode
                    # cyborg.env.env.tracker.render()

                    blackboard.test_counter = 0
                    blackboard.step = 0

                    # subtract 3 because of setup steps
                    for j in range(num_steps):
                        #print(j)
                        root.tick_once()
                        blackboard.step += 1
                        red_observation=blackboard.cyborg.get_observation('Red')
                        blue_outcome = blackboard.observation
                        blue_action= blackboard.cyborg.get_last_action('Blue')
                        red_action= blackboard.cyborg.get_last_action('Red')
                        #print(blackboard.cyborg.get_last_action("Red"))
                        #print(blackboard.cyborg.get_last_action("Blue"))
                        #print(blackboard.observation)
                        #print(blackboard.cyborg.get_observation("Red"))
                        #print(blackboard.cyborg.get_last_action("Red")) 
                        
                    agent.end_episode()
                    total_reward.append(sum(blackboard.r))
                    actions.append(blackboard.a)
                    # observation = cyborg.reset().observation
                    blackboard.observation = blackboard.wrapped_cyborg.reset()
                    #print("ep done. reward is: ", sum(blackboard.r))
           
            #print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            #with open(file_name, 'a+') as data:
             #   data.write(f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
             #   for act, sum_rew in zip(actions, total_reward):
              #      data.write(f'actions: {act}, total reward: {sum_rew}\n')
