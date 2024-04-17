# Dynamic Three-phase Parallel Deep Q-Network (DTPDQN)

## Dynamic Multi-Objective Flexible Job Shop Scheduling: A Three-Phase Deep Reinforcement Learning Approach


This repo has three python files: DQN.py, Job_shop.py, and Object_for_FJSP.py. The following is a brief explanation about how the code is linked to our work.

- DQN.py is the main running code. In this file, we mainly construct the dynamic multi-objective double deep Q-learning network (DTPDQN), and provide the instance generation algorithm (the function `Instance_Generator` in code). The uncertain events, including Job Insertion, Job Delivery Deadline Modification, Job Cancellation, Job Operation Modification, and Machine Addition are reflected in this file. And the three optimization objectives are also calculated in this file (the function `main` in code).

- Job_shop.py contains feature calculation function (the function `Features` in code) to extract five states features as the input of DTPDQN. Five job dispatching rules (the function `job_rule1--job_rule5` in code), three machine dispatching rules (the function `machine_rule1--machine_rule5` in code) and three reward functions (the functions `reward1`, `reward2`, `reward3` in code) are provided in this file. It also has a scheduling function (the function  `scheduling` in code) to assign jobs and machines. Where the other two uncertain events of Machine Breakdown and Machine Switching are embedded in this function scheduling.

- Object_for_FJSP.py is the class to storage the scheduling information of scheduled jobs and machines. And it also provides a local search algorithm (the function `idle_time` in code) to further optimize the result of DTPDQN (LSDTPDQN).


## [Requirement]
- Python >= 3.x
- Numpy https://numpy.org/
- Keras https://keras.io/
