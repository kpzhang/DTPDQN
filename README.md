# Dynamic Three-phase Parallel Deep Q-Network (DTPDQN)

## Dynamic Multi-Objective Flexible Job Shop Scheduling: A Three-Phase Deep Reinforcement Learning Approach


The project contains three python files: DQN, job_shop and Object_for_FJSP. The following is a brief explanation about how the code is linked to the content of our work.

- DQN.py is the main running file in the project. In this file, we mainly construct the dynamic multi-objective double deep Q-learning network (DTPDQN in paper), and provide the instance generation algorithm (function Instance_Generator in code). The uncertain events of Job Insertion, Job Delivery Deadline Modification, Job Cancellation, Job Operation Modification, Machine Addition are reflected in this file. And the three objectives are calculated in this file (function main in code).

- job_shop.py contains feature calculation function (function Features in code) to extract five states features as the input of DTPDQN. Five job dispatching rules (function job_rule1--job_rule5 in code), three machine dispatching rules (function machine_rule1--machine_rule5 in code) and three reward functions (function reward1, reward2, reward3 in code) are provided in this file. Then it also contains a scheduling function (function scheduling in code) to assign jobs and machines. Where the other two uncertain events of Machine Breakdown and Machine Switching are embedded in this function scheduling.

- Object_for_FJSP.py is the class to storage the scheduling information of scheduled jobs and machines. And it also provides a local search function (function idle_time in code) to further optimize the result of DTPDQN (LSDTPDQN in paper).
