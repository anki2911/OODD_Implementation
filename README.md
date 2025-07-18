# Out-of-Distribution Detection for any RL agents with discrete states and discrete actions (Implementation)
Implementation of "Out-of-Distribution Detection for Neurosymbolic Autonomous Cyber Agents" https://ieeexplore.ieee.org/document/10849024

In this work, we develop an out-of-distribution (OOD) Monitoring algorithm that uses a Probabilistic Neural Network (PNN) to detect anomalous or OOD situations of RL-based agents with discrete states and discrete actions. To demonstrate the effectiveness of the proposed approach, we integrate the OOD monitoring algorithm with a neurosymbolic autonomous cyber agent that uses behavior trees with learning-enabled components. We evaluate the proposed approach in a simulated cyber environment under different adversarial strategies. 

For more details about the Out-of-Distribution implementation, please refer to our publication 
Ankita Samaddar, Nicholas Potteiger, Xenofon Koutsoukos, "Out-of-Distribution Detection for Neurosymbolic Autonomous Cyber Agents", IEEE International Conference on AI and Cybersecurity (ICAIC), 2025 which can be referenced as 

@INPROCEEDINGS{OODD,<br>
  author={Samaddar, Ankita and Potteiger, Nicholas and Koutsoukos, Xenofon},<br>booktitle={2025 IEEE 4th International Conference on AI in Cybersecurity (ICAIC)},<br>
  title={Out-of-Distribution Detection for Neurosymbolic Autonomous Cyber Agents},<br>year={2025},<br>doi={10.1109/ICAIC63015.2025.10849024}}

# Steps to use the codebase

1. Install CybORG CAGE-Challenge Scenario 2 from https://github.com/cage-challenge/cage-challenge-2 following the instructions in https://github.com/cage-challenge/cage-challenge-2/blob/main/CybORG/README.md
   
2. To facilitate strategy switching, download the modified version of CybORG that allows strategy switching from https://github.com/anki2911/Cage_Challenge_2_StrategySwitch and follow the instructions in Step 1 
   
3. Run OOD tests with CybORG using evaluation_bt_nodes_with_OOD_main.py. Run evaluation_bt_nodes_with_OOD.py if you want to run Strategy Switching to test different Red Agent strategies.
   
4. Alternatively, you can generate your own dataset by changing the parameters in Dataset_Generation.py; or use existing datasets by extracting the datasets inside cage-2-ebt/Models/Dataset_train/"adversarial strategy" where "adversarial strategy" can be either B_line or Meander. You can run the OOD tests offline without the simulator in the loop by using OOD_TEST.py.

5. The OOD_TEST.py and PNN_TEST.py files can be used to detect OOD situations for any RL based agent with discrete states and discrete actions by incorporating different datasets to train the PNN.
