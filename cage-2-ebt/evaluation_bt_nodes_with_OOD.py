import py_trees_devel.py_trees as py_trees
import numpy as np
import copy
from Agents.PPOAgent import PPOAgent
import torch
import torch.nn as nn
import loaddata

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ChangeStratCheck(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Change Strategy?"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "step", access = py_trees.common.Access.WRITE)

        self.blackboard.register_key(key = "switch", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "window", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "lstm_model", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "need_switch", access = py_trees.common.Access.WRITE)
    
    def update(self):
        print("Inside Change Strategy Check")
        
        #  or self.blackboard.step == self.blackboard.switch.switch_step
        if self.blackboard.step == 3 or self.blackboard.step == self.blackboard.switch.switch_step:
            # print("switch", self.blackboard.step)
            return py_trees.common.Status.FAILURE
        # elif self.blackboard.need_switch and len(self.blackboard.window) == 5:
        #     tens_window = torch.tensor(self.blackboard.window, dtype=torch.float32).unsqueeze(0).to(device)
        #     out = self.blackboard.lstm_model(tens_window)
        #     if (out > 0.5).item():
        #         # print("switch")
        #         self.blackboard.need_switch = False
        #         return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class ChangeStrat(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Change Strategy"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "scan_state", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "observation", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "step", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "OOD_Model", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "PNN_Meander", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "PNN_Bline", access = py_trees.common.Access.WRITE)

    def update(self):
        print("Change Strategy")
        
        if self.blackboard.step == 3:
            scan_state_copy = copy.copy(self.blackboard.agent.scan_state)
            # print(self.blackboard.agent.scan_state)
            
            self.blackboard.agent.add_scan(self.blackboard.observation)

            if self.blackboard.agent.fingerprint_meander():
                self.blackboard.agent.agent = self.blackboard.agent.load_meander()
                self.blackboard.OOD_Model = self.blackboard.PNN_Meander
            elif self.blackboard.agent.fingerprint_bline():
                self.blackboard.agent.agent = self.blackboard.agent.load_bline()
                self.blackboard.OOD_Model = self.blackboard.PNN_Bline
            else:
                self.blackboard.agent.agent = self.blackboard.agent.load_sleep()
        
            #print(self.blackboard.agent.agent)
            # add decoys and scan state
            self.blackboard.agent.agent.current_decoys = {1000: [55], # enterprise0
                                                    1001: [], # enterprise1
                                                    1002: [], # enterprise2
                                                    1003: [], # user1
                                                    1004: [51, 116], # user2
                                                    1005: [], # user3
                                                    1006: [], # user4
                                                    1007: [], # defender
                                                    1008: []} # opserver0
            # add old since it will add new scan in its own action (since recieves latest observation)
            self.blackboard.agent.agent.scan_state = scan_state_copy
            self.blackboard.agent.agent_loaded = True
        else:
            # print("bline loaded")
            self.blackboard.agent.agent = self.blackboard.agent.load_bline()
            self.blackboard.OOD_Model = self.blackboard.PNN_Bline

        return py_trees.common.Status.SUCCESS


class GetPPOAction(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Get PPO Action"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "observation", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "action_space", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "step", access = py_trees.common.Access.WRITE)


    def update(self):
        print("Inside Get PPO Action")
        
        #self.blackboard.action = self.agent.get_action(self.blackboard.observation)

        self.blackboard.agent.agent.add_scan(self.blackboard.observation)
        self.blackboard.observation = self.blackboard.agent.agent.pad_observation(self.blackboard.observation)
        state = torch.FloatTensor(self.blackboard.observation.reshape(1, -1)).to(device)
        action = self.blackboard.agent.agent.old_policy.act(state, self.blackboard.agent.agent.memory,
                                                            deterministic = self.blackboard.agent.agent.deterministic)
        self.blackboard.action = self.blackboard.action_space[action]

        return py_trees.common.Status.SUCCESS


class AnalyzeCheck(py_trees.behaviour.Behaviour):
    
    def __init__(self, name: str = "Action is Analyze?"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)

    def update(self):
        print("Inside Analyze Check")
        
        if self.blackboard.action < 15:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class Analyze(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Analyze"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "transformed_action", access = py_trees.common.Access.WRITE)

    def update(self):
        print("Inside Analyze")
        
        self.blackboard.transformed_action = self.blackboard.action
        return py_trees.common.Status.SUCCESS


class DeployDecoyCheck(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Action is Deploy Decoy?"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "decoy_ids", access = py_trees.common.Access.WRITE)

    def update(self):
        print("Inside Deploy Decoy Check")
        
        # print(self.blackboard.action)
        if self.blackboard.action in self.blackboard.decoy_ids:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class DeployDecoy(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Deploy Decoy"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "transformed_action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "observation", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "decoy_ids", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "action_space", access = py_trees.common.Access.WRITE)

    def update(self):
        print("Inside Deploy Decoy")
        
        host = self.blackboard.action
        try:
        # pick the top remaining decoy
            self.blackboard.transformed_action = [a for a in self.blackboard.agent.agent.greedy_decoys[host]
                                                  if a not in self.blackboard.agent.agent.current_decoys[host]][0]
            self.blackboard.agent.agent.add_decoy(self.blackboard.transformed_action, host)
        except:
            state = torch.FloatTensor(self.blackboard.observation.reshape(1, -1)).to(device)
            actions = self.blackboard.agent.agent.old_policy.act(state, self.blackboard.agent.agent.memory, full=True)
            max_actions = torch.sort(actions, dim=1, descending=True)
            max_actions = max_actions.indices
            max_actions = max_actions.tolist()

            # don't need top action since already know it can't be used (hence could put [1:] here, left for clarity)
            for action_ in max_actions[0]:
                a = self.blackboard.action_space[action_]
                # if next best action is decoy, check if its full also
                if a in self.blackboard.agent.agent.current_decoys.keys():
                    if len(self.blackboard.agent.agent.current_decoys[a]) < len(self.blackboard.agent.agent.greedy_decoys[a]):
                        self.blackboard.transformed_action = self.blackboard.agent.agent.select_decoy(a, self.blackboard.observation)
                        self.blackboard.agent.agent.add_decoy(self.blackboard.transformed_action, a)
                        break
                else:
                    # don't select a next best action if "restore", likely too aggressive for 30-50 episodes
                    if a not in self.blackboard.agent.agent.restore_decoy_mapping.keys():
                        self.blackboard.transformed_action = a
                        break
        
        return py_trees.common.Status.SUCCESS


class RemoveDecoysCheck(py_trees.behaviour.Behaviour):
    
    def __init__(self, name: str = "Action is Restore Host?"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)

    def update(self):
        print("Inside Remove Decoy Check")
        
        if self.blackboard.action in self.blackboard.agent.agent.restore_decoy_mapping.keys():
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class RemoveDecoys(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Remove Decoys"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "transformed_action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "agent", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "decoy_ids", access = py_trees.common.Access.WRITE)

    def update(self):
        print("Inside Remove Decoy")
        
        self.blackboard.transformed_action = self.blackboard.action
        if self.blackboard.action > 27:
            for decoy in self.blackboard.agent.agent.restore_decoy_mapping[self.blackboard.action]:
                for host in self.blackboard.decoy_ids:
                    if decoy in self.blackboard.agent.agent.current_decoys[host]:
                        self.blackboard.agent.agent.current_decoys[host].remove(decoy)

        return py_trees.common.Status.SUCCESS


class DetectIDCheck(py_trees.behaviour.Behaviour):
    def __init__(self, name: str = "Detect ID Check"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "transformed_action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "prev_observation", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "cur_observation", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "observation", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "PNN_Bline_PrevState_Labels", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "PNN_Bline_CurState_Labels", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "PNN_Bline_Action_Labels", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "PNN_Meander_PrevState_Labels", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "PNN_Meander_CurState_Labels", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "PNN_Meander_Action_Labels", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "OOD_Model", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "PNN_Meander", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "PNN_Bline", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "Distribution_Status", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "freq", access = py_trees.common.Access.WRITE)
        
    def update(self):
        print("Inside ID Check")
        #print(self.blackboard.prev_observation)
        #print(self.blackboard.observation)
        #print(self.blackboard.transformed_action)
        
        prev_state = []
        cur_state = []
        prev_action = []
        prev_action.append(self.blackboard.transformed_action)
        #print(len(self.blackboard.prev_observation))
        for i in range(len(self.blackboard.prev_observation)):
            prev_state.append(self.blackboard.prev_observation[i])
            cur_state.append(self.blackboard.cur_observation[i])
        self.blackboard.freq = 0.0
        if self.blackboard.OOD_Model == self.blackboard.PNN_Meander:
            print("PNN_Meander")
            S_test_labelled = loaddata.get_label_NS(prev_state,self.blackboard.PNN_Meander_PrevState_Labels)
            NS_test_labelled = loaddata.get_label_NS(cur_state,self.blackboard.PNN_Meander_CurState_Labels) 
            A_test_labelled = loaddata.get_label_action(prev_action,self.blackboard.PNN_Meander_Action_Labels)
            if S_test_labelled > 0 and NS_test_labelled > 0 and A_test_labelled > 0:
                test_data_sample = []
                test_data_sample.append(S_test_labelled)
                test_data_sample.append(A_test_labelled)
                test_data_sample.append(NS_test_labelled)
                print(test_data_sample)
                status, prob = self.blackboard.PNN_Meander.test(test_data_sample[0:2],test_data_sample[2:])
                self.blackboard.freq = prob
                if status == -1:
                    #Transition to output state not in training data
                    print("No Transition to Next State")
                    self.blackboard.Distribution_Status = 1
                    return py_trees.common.Status.SUCCESS
                elif status == -2:
                    #Misprediction to different output state
                    print("Transition to output state not in training data")
                    self.blackboard.Distribution_Status = 1                   
                    return py_trees.common.Status.SUCCESS
                else:
                    print("In Distribution")
                    self.blackboard.Distribution_Status = 1
                    return py_trees.common.Status.SUCCESS
            else:
                if NS_test_labelled == 0:
                    #Current state not in dataset
                    print("Current State is not in training data")
                    self.blackboard.Distribution_Status = 0
                    return py_trees.common.Status.FAILURE
                else:
                    #Action not in dataset
                    print("Action is not in training data")  
                    self.blackboard.Distribution_Status = 1
                    return py_trees.common.Status.SUCCESS      
            
        elif self.blackboard.OOD_Model == self.blackboard.PNN_Bline:
            print("PNN_Bline")
            S_test_labelled = loaddata.get_label_NS(prev_state,self.blackboard.PNN_Bline_PrevState_Labels)
            NS_test_labelled = loaddata.get_label_NS(cur_state,self.blackboard.PNN_Bline_CurState_Labels) 
            A_test_labelled = loaddata.get_label_action(prev_action,self.blackboard.PNN_Bline_Action_Labels)
            if S_test_labelled > 0 and NS_test_labelled > 0 and A_test_labelled > 0:
                test_data_sample = []
                test_data_sample.append(S_test_labelled)
                test_data_sample.append(A_test_labelled)
                test_data_sample.append(NS_test_labelled)
                status, prob = self.blackboard.PNN_Bline.test(test_data_sample[0:2],test_data_sample[2:])
                self.blackboard.freq = prob
                if status == -1:
                    #Transition to output state not in training data
                    print("No Transition to Next State")
                    self.blackboard.Distribution_Status = 1
                    return py_trees.common.Status.SUCCESS
                elif status == -2:
                    #Misprediction to different output state
                    print("Transition to output state not in training data")
                    self.blackboard.Distribution_Status = 1
                    return py_trees.common.Status.SUCCESS
                else:
                    print("In Distribution")
                    self.blackboard.Distribution_Status = 1
                    return py_trees.common.Status.SUCCESS
            else:
                if NS_test_labelled == 0:
                    #Current state not in dataset
                    print("Current State is not in training data")
                    self.blackboard.Distribution_Status = 0
                    return py_trees.common.Status.FAILURE
                else:
                    #Action not in dataset
                    print("Action is not in training data")
                    self.blackboard.Distribution_Status = 1
                    return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.SUCCESS
    
class GetSafeAction(py_trees.behaviour.Behaviour):
    def __init__(self, name: str = "Get Safe Action"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "transformed_action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "prev_observation", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "cur_observation", access = py_trees.common.Access.WRITE)
        
    def update(self):
        print("Inside Get Safe Action")
        #return py_trees.common.Status.SUCCESS

        l = self.blackboard.cur_observation
        l = l.reshape(len(l),1)
        i = 13
        while (i >= 1):
            k = (i-1)*4
            p1 = l[k]
            p2 = l[k+1]
            p3 = l[k+2]
            p4 = l[k+3]
    
            if i == 1:              
                if p3 == 0 and p4 == 1:
                    print("Scanned Defender")
                    #self.blackboard.transformed_action = 28
                    self.blackboard.transformed_action = 132
                elif p3 == 1 and p4 == 1:
                    print("Exploited Defender")  
                    self.blackboard.transformed_action = 132 
                    #if p1 == 0 and p2 == 1:
                    #    print("User Level Exploit")
                    #    self.blackboard.transformed_action = 15
                    #elif p1 == 1 and p2 == 1:
                    #    print("System Level Exploit")
                    #    self.blackboard.transformed_action = 132   
            elif i == 2:
                if p3 == 0 and p4 == 1:
                    print("Scanned Enterprise 0")
                    self.blackboard.transformed_action = 133
                    #self.blackboard.transformed_action = 29
                elif p3 == 1 and p4 == 1:
                    print("Exploited Enterprise 0")
                    self.blackboard.transformed_action = 133
                    #if p1 == 0 and p2 == 1:
                    #    print("User Level Exploit")
                    #    self.blackboard.transformed_action = 16
                    #elif p1 == 1 and p2 == 1:
                    #    print("System Level Exploit")
                    #    self.blackboard.transformed_action = 133  
            elif i == 3:
                if p3 == 0 and p4 == 1:
                    print("Scanned Enterprise 1")
                    self.blackboard.transformed_action = 134 
                    #self.blackboard.transformed_action = 30
                elif p3 == 1 and p4 == 1:
                    print("Exploited Enterprise 1")
                    self.blackboard.transformed_action = 134
                    #if p1 == 0 and p2 == 1:
                    #    print("User Level Exploit")
                    #    self.blackboard.transformed_action = 17
                    #elif p1 == 1 and p2 == 1:
                    #    print("System Level Exploit")
                    #    self.blackboard.transformed_action = 134    
            elif i == 4:
                if p3 == 0 and p4 == 1:
                    print("Scanned Enterprise 2")
                    self.blackboard.transformed_action = 135
                    #self.blackboard.transformed_action = 31
                elif p3 == 1 and p4 == 1:
                    print("Exploited Enterprise 2") 
                    self.blackboard.transformed_action = 135
                    #if p1 == 0 and p2 == 1:
                    #    print("User Level Exploit")
                    #    self.blackboard.transformed_action = 18
                    #elif p1 == 1 and p2 == 1:
                    #    print("System Level Exploit")
                    #    self.blackboard.transformed_action = 135  
            elif i == 5:
                if p3 == 0 and p4 == 1:
                    print("Scanned Operational Host 0")
                    self.blackboard.transformed_action = 136
                    #self.blackboard.transformed_action = 32
                elif p3 == 1 and p4 == 1:
                    print("Exploited Operational Host 0")
                    self.blackboard.transformed_action = 136
                    #if p1 == 0 and p2 == 1:
                    #    print("User Level Exploit")
                    #    self.blackboard.transformed_action = 19
                    #elif p1 == 1 and p2 == 1:
                    #    print("System Level Exploit")
                    #    self.blackboard.transformed_action = 136     
            elif i == 6:
                if p3 == 0 and p4 == 1:
                    print("Scanned Operational Host 1")
                    self.blackboard.transformed_action = 137
                    #self.blackboard.transformed_action = 33
                elif p3 == 1 and p4 == 1:
                    print("Exploited Operational Host 1") 
                    self.blackboard.transformed_action = 137
                    #if p1 == 0 and p2 == 1:
                    #    print("User Level Exploit")
                    #    self.blackboard.transformed_action = 20
                    #elif p1 == 1 and p2 == 1:
                    #    print("System Level Exploit")
                    #    self.blackboard.transformed_action = 137   
            elif i == 7:
                if p3 == 0 and p4 == 1:
                    print("Scanned Operational Host 2")
                    self.blackboard.transformed_action = 138
                    #self.blackboard.transformed_action = 34
                elif p3 == 1 and p4 == 1:
                    print("Exploited Access Operational Host 2") 
                    self.blackboard.transformed_action = 138
                    #if p1 == 0 and p2 == 1:
                    #    print("User Level Exploit")
                    #    self.blackboard.transformed_action = 21
                    #elif p1 == 1 and p2 == 1:
                    #    print("System Level Exploit")
                    #    self.blackboard.transformed_action = 138 
            elif i == 8:
                if p3 == 0 and p4 == 1:
                    print("Scanned Op_Server0")
                    self.blackboard.transformed_action = 139
                    #self.blackboard.transformed_action = 35
                elif p3 == 1 and p4 == 1:
                    print("Exploited Op_Server 0") 
                    self.blackboard.transformed_action = 139
                    #if p1 == 0 and p2 == 1:
                    #    print("User Level Exploit")
                    #    self.blackboard.transformed_action = 22
                    #elif p1 == 1 and p2 == 1:
                    #    print("System Level Exploit")
                    #    self.blackboard.transformed_action = 139 
            elif i == 9:
                if p3 == 0 and p4 == 1:
                    print("Scanned User0")
                    self.blackboard.transformed_action = 140
                    #self.blackboard.transformed_action = 36
                elif p3 == 1 and p4 == 1:
                    print("Exploited User 0")
                    self.blackboard.transformed_action = 140
                    #if p1 == 0 and p2 == 1:
                    #    print("User Level Exploit")
                    #    self.blackboard.transformed_action = 23
                    #elif p1 == 1 and p2 == 1:
                    #    print("System Level Exploit")
                    #    self.blackboard.transformed_action = 140
            elif i == 10:
                if p3 == 0 and p4 == 1:
                    print("Scanned User1")
                    self.blackboard.transformed_action = 141
                    #self.blackboard.transformed_action = 37
                elif p3 == 1 and p4 == 1:
                    print("Exploited Access User 1")
                    self.blackboard.transformed_action = 141
                    #if p1 == 0 and p2 == 1:
                    #    print("User Level Exploit")
                    #    self.blackboard.transformed_action = 24
                    #elif p1 == 1 and p2 == 1:
                    #    print("System Level Exploit")
                    #    self.blackboard.transformed_action = 141
            elif i == 11:
                if p3 == 0 and p4 == 1:
                    print("Scanned User2")
                    self.blackboard.transformed_action = 142
                    #self.blackboard.transformed_action = 38
                elif p3 == 1 and p4 == 1:
                    print("Exploited User 2")
                    self.blackboard.transformed_action = 142
                    #if p1 == 0 and p2 == 1:
                    #    print("User Level Exploit")
                    #    self.blackboard.transformed_action = 25
                    #elif p1 == 1 and p2 == 1:
                    #    print("System Level Exploit")
                    #    self.blackboard.transformed_action = 142 
            elif i == 12:
                if p3 == 0 and p4 == 1:
                    print("Scanned User3")
                    self.blackboard.transformed_action = 143
                    #self.blackboard.transformed_action = 39
                elif p3 == 1 and p4 == 1:
                    print("Exploited User 3")
                    self.blackboard.transformed_action = 143
                    #if p1 == 0 and p2 == 1:
                    #    print("User Level Exploit")
                    #    self.blackboard.transformed_action = 26
                    #elif p1 == 1 and p2 == 1:
                    #    print("System Level Exploit")
                    #    self.blackboard.transformed_action = 143
            if i == 13:
                if p3 == 0 and p4 == 1:
                    print("Scanned User4")
                    self.blackboard.transformed_action = 144
                    #self.blackboard.transformed_action = 40
                elif p3 == 1 and p4 == 1:
                    print("Exploited User 4")
                    self.blackboard.transformed_action = 144
                    #if p1 == 0 and p2 == 1:
                    #    print("User Level Exploit")
                    #    self.blackboard.transformed_action = 27
                    #elif p1 == 1 and p2 == 1:
                    #    print("System Level Exploit")
                    #    self.blackboard.transformed_action = 144
                                             
            i = i - 1

        return py_trees.common.Status.SUCCESS
    
class Is_OOD(py_trees.behaviour.Behaviour):
    def __init__(self, name: str = "Get Safe Action"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "Distribution_Status", access = py_trees.common.Access.WRITE)
        
    def update(self):
        print("Inside Is OOD")
        if self.blackboard.Distribution_Status == 0:
            print("OOD")
            return py_trees.common.Status.SUCCESS
            #return py_trees.common.Status.FAILURE
        else:
            print("In Distribution")
            return py_trees.common.Status.FAILURE

class ExecuteActions(py_trees.behaviour.Behaviour):

    def __init__(self, name: str = "Execute Actions"):
        super().__init__(name = name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key = "observation", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "wrapped_cyborg", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "action", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "r", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "a", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "cyborg", access = py_trees.common.Access.WRITE)
        self.blackboard.register_key(key = "step", access = py_trees.common.Access.WRITE)

        self.blackboard.register_key(key = "test_counter", access = py_trees.common.Access.WRITE)

    def update(self):
        # print(self.blackboard.observation)
        self.blackboard.observation, reward, done, info = self.blackboard.wrapped_cyborg.step(self.blackboard.action)
        # print(self.blackboard.action)
        self.blackboard.r.append(reward)
        self.blackboard.a.append((str(self.blackboard.cyborg.get_last_action('Blue')),
                                  str(self.blackboard.cyborg.get_last_action('Red'))))
        
        self.blackboard.test_counter += 1
        #print(self.blackboard.test_counter)

        return py_trees.common.Status.SUCCESS
    