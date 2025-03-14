# OODD_Implementation
Implementation of "Out-of-Distribution Detection for Neurosymbolic Autonomous Cyber Agents" https://ieeexplore.ieee.org/document/10849024

In this work, we develop an out-of-distribution (OOD) Monitoring algorithm that uses a Probabilistic Neural Network (PNN) to detect anomalous or OOD situations of RL-based agents with discrete states and discrete actions. To demonstrate the effectiveness of the proposed approach, we integrate the OOD monitoring algorithm with a neurosymbolic autonomous cyber agent that uses behavior trees with learning-enabled components. We evaluate the proposed approach in a simulated cyber environment under different adversarial strategies. Experimental results over a large number of episodes illustrate the overall efficiency of our proposed approach.

# Steps to use the codebase

1. Install CybORG CAGE-Challenge Scenario 2 from https://github.com/cage-challenge/cage-challenge-2 following the instructions in https://github.com/cage-challenge/cage-challenge-2/blob/main/CybORG/README.md
2. Run OOD tests with CybORG using evaluation_with_OOD_main.py.
3. Alternatively, you can generate your own dataset by changing the parameters in Dataset_Generation.py; or use existing datasets by extracting the datasets inside cage-2-ebt/Models/Dataset_train/"adversarial strategy" depending on the strategy you want to work with. You can run the OOD tests offline without the simulator in the loop by using OOD_TEST.py.  
