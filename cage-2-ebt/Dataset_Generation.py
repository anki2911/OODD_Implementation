import torch
import numpy as np
import os
from Agents.PPOAgent import PPOAgent
import random
from datetime import datetime
import subprocess
import inspect
import time
from statistics import mean, stdev
from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.MainAgent import MainAgent

MAX_EPS = 1
agent_name = 'Blue'
random.seed(0)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

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

    # Change this line to load your agent
    agent = MainAgent()

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')


    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    folder = 'Dataset_train'
    ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    Act = []
    Red_Act = []
    State = []
                
    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [10]:
        for red_agent in [RedMeanderAgent, B_lineAgent]:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)

            observation = wrapped_cyborg.reset()
            # observation = cyborg.reset().observation
            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)
            
            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                IP = cyborg.get_ip_map()
                subnet_map = cyborg.get_cidr_map()
                #print(subnet_map)
                #print(subnet_map)
                r = []
                a = []
                o = []
                #if red_agent == B_lineAgent:
                    #print("Action Space of BlineAgent")
                    #print(wrapped_cyborg.get_action_space('Red'))
                #    f = open(str(ckpt_folder) + "/bline/Steps_addn_1_" + str(num_steps) + "/D_bline_" + str(num_steps) + "_" + str(i) + ".txt","w")
                #    g = open(str(ckpt_folder) + "/bline/Steps_addn_1_" + str(num_steps) + "/A_bline_" + str(num_steps) + "_" + str(i) + ".txt","w")
                #    h = open(str(ckpt_folder) + "/bline/Steps_addn_1_" + str(num_steps) + "/R_bline_" + str(num_steps) + "_" + str(i) + ".txt","w")
                #    red_act = open(str(ckpt_folder) + "/bline/Steps_addn_1_" + str(num_steps) + "/RA_bline_" + str(num_steps) + "_" + str(i) + ".txt","w")
                #else:
                    #print("Action Space of RedMeander Agent")
                    #print(wrapped_cyborg.get_action_space('Red'))
                #    f = open(str(ckpt_folder) + "/meander/Steps_addn_1_" + str(num_steps) + "/D_meander_" + str(num_steps) + "_" + str(i) + ".txt","w")
                #    g = open(str(ckpt_folder) + "/meander/Steps_addn_1_" + str(num_steps) + "/A_meander_" + str(num_steps) + "_" + str(i) + ".txt","w")
                #    h = open(str(ckpt_folder) + "/meander/Steps_addn_1_" + str(num_steps) + "/R_meander_" + str(num_steps) + "_" + str(i) + ".txt","w")
                #    red_act = open(str(ckpt_folder) + "/meander/Steps_addn_1_" + str(num_steps) + "/RA_meander_" + str(num_steps) + "_" + str(i) + ".txt","w")
                
                #for i in range(len(observation)):
                #    #print(observation[i])
                #    f.write(str(observation[i]))
                #    f.write(" ")
                #f.write("\n")
                
                # cyborg.env.env.tracker.render()
                j = 0
                for j in range(num_steps):
                    action = agent.get_action(observation, action_space)
                    #if j == 0:
                        #for epo in range(len(observation)):
                            #print(observation[i])
                            #f.write(str(observation[epo]))
                            #f.write(" ")
                        #f.write("\n")
                    if str(observation) not in State:
                        State.append(str(observation))

                    #print(action)
                    observation, rew, done, info = wrapped_cyborg.step(action)
                    #print(observation)
                    
                    
                    if action not in Act:
                        Act.append(action)
                        #print(action)
                    ra = wrapped_cyborg.get_last_action('Red')
                    ba = wrapped_cyborg.get_last_action('Blue')
                    
                    if ra not in Red_Act:
                        Red_Act.append(ra)

                    print (ra)
                    #if j > -1:                        
                    for keys in IP.keys():
                        if str(IP[keys]) in str(ra):
                            intermediate_form = str(ra)
                            intermediate_form = intermediate_form.replace(str(IP[keys]),str(keys))
                            ra = intermediate_form
                        else:
                            pass  
                        #print(str(ra))
                        #print(intermediate_form)
                    for keys in subnet_map.keys():
                        if str(subnet_map[keys]) in str(ra):
                            intermediate_form = str(ra)
                            intermediate_form = intermediate_form.replace(str(subnet_map[keys]),str(keys))
                            ra = intermediate_form
                        else:
                            pass  
                        
                    
                    #if j > -1:
                    #    g.write(str(action))
                    #    g.write("\n")
                    #    h.write(str(float(rew)))
                    #    h.write("\n")
                    #    if str(ra) == str(intermediate_form):
                    #        red_act.write(str(intermediate_form))
                    #    else:
                    #        red_act.write(str(ra))
                    #    red_act.write("\n")
                        
                        

                    #    for epo in range(len(observation)):
                            #print(observation[i])
                    #        f.write(str(observation[epo]))
                    #        f.write(" ")
                    #    f.write("\n")
                    
                    
                    o.append(observation)
                    #print(observation)
                    #print(rew)
                    # result = cyborg.step(agent_name, action)
                    r.append(rew)
                    # r.append(result.reward)
                    #a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
                    #print(cyborg.get_last_action('Red'))
                    #print(cyborg.get_last_action('Blue'))
                    #print(agent.agent)
                
                agent.end_episode()
                total_reward.append(sum(r))
                #print(total_reward)
                #actions.append(a)
                #print(actions)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()

                #print(Act)
                #print(Red_Act)
                #print(red_agent)
                #print(i)
    #print(len(State))
    #print(State)
    #print(len(Act))