import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import random
from collections import deque
from tensorflow.keras import layers,models
from Job_shop import Situation
from tensorflow.keras.optimizers import Adam
 
 
class DQN:
    def __init__(self,Hid_Size1,Hid_Size2,Hid_Size3):
        self.Hid_Size1 = Hid_Size1
        self.Hid_Size2 = Hid_Size2
        self.Hid_Size3 = Hid_Size3
 
        # ------------Hidden layer=2  30 nodes each layer--------------
        model = models.Sequential()
        model.add(layers.Input(shape=(5,)))
        model.add(layers.Dense(self.Hid_Size1, name='l1'))
        model.add(layers.Dense(self.Hid_Size1, name='l2'))
        # model.add(layers.Dense(self.Hid_Size1, name='l3'))
        # model.add(layers.Dense(self.Hid_Size1, name='l4'))
        # model.add(layers.Dense(self.Hid_Size1, name='l5'))
        model.add(layers.Dense(3, name='l6'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.001))
        # ------------Hidden layer=2 30 nodes each layer--------------
        model1 = models.Sequential()
        model1.add(layers.Input(shape=(6,)))
        model1.add(layers.Dense(self.Hid_Size2, name='l1'))
        model1.add(layers.Dense(self.Hid_Size2, name='l2'))
        # model1.add(layers.Dense(self.Hid_Size2, name='l3'))
        # model1.add(layers.Dense(self.Hid_Size2, name='l4'))
        # model1.add(layers.Dense(self.Hid_Size2, name='l5'))
        model1.add(layers.Dense(5, name='l6'))
        model1.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.001))
        # ------------Hidden layer=2 30 nodes each layer--------------
        model2 = models.Sequential()
        model2.add(layers.Input(shape=(7,)))
        model2.add(layers.Dense(self.Hid_Size3, name='l1'))
        model2.add(layers.Dense(self.Hid_Size3, name='l2'))
        # model2.add(layers.Dense(self.Hid_Size3, name='l3'))
        # model2.add(layers.Dense(self.Hid_Size3, name='l4'))
        # model2.add(layers.Dense(self.Hid_Size3, name='l5'))
        model2.add(layers.Dense(3, name='l6'))
        model2.compile(loss='mse',
                       optimizer=Adam(learning_rate=0.001))
        # # model.summary()
        self.model = model
        self.model1 = model1
        self.model2 = model2
 
        #------------Q-network Parameters-------------
        self.gama = 0.95  # γ经验折损率
        # self.lr = 0.001  # 学习率
        self.global_step = 0
        self.update_target_steps = 200  # 更新目标网络的步长
        self.target_model = self.model
        self.target_model1 = self.model1
        self.target_model2 = self.model2

        # -------------------Agent-------------------
        self.e_greedy = 0.6
        self.e_greedy_decrement = 0.0001
        self.L = 20  # Number of training episodes L

        # ---------------Replay Buffer---------------
        self.buffer = deque(maxlen=2000)
        self.Batch_size = 8  # Batch Size of Samples to perform gradient descent
 
    def replace_target(self):
        self.target_model.get_layer(name='l1').set_weights(self.model.get_layer(name='l1').get_weights())
        self.target_model.get_layer(name='l2').set_weights(self.model.get_layer(name='l2').get_weights())
        # self.target_model.get_layer(name='l3').set_weights(self.model.get_layer(name='l3').get_weights())
        # self.target_model.get_layer(name='l4').set_weights(self.model.get_layer(name='l4').get_weights())
        # self.target_model.get_layer(name='l5').set_weights(self.model.get_layer(name='l5').get_weights())
        self.target_model.get_layer(name='l6').set_weights(self.model.get_layer(name='l6').get_weights())

        self.target_model1.get_layer(name='l1').set_weights(self.model1.get_layer(name='l1').get_weights())
        self.target_model1.get_layer(name='l2').set_weights(self.model1.get_layer(name='l2').get_weights())
        # self.target_model1.get_layer(name='l3').set_weights(self.model1.get_layer(name='l3').get_weights())
        # self.target_model1.get_layer(name='l4').set_weights(self.model1.get_layer(name='l4').get_weights())
        # self.target_model1.get_layer(name='l5').set_weights(self.model1.get_layer(name='l5').get_weights())
        self.target_model1.get_layer(name='l6').set_weights(self.model1.get_layer(name='l6').get_weights())

        self.target_model2.get_layer(name='l1').set_weights(self.model2.get_layer(name='l1').get_weights())
        self.target_model2.get_layer(name='l2').set_weights(self.model2.get_layer(name='l2').get_weights())
        # self.target_model2.get_layer(name='l3').set_weights(self.model2.get_layer(name='l3').get_weights())
        # self.target_model2.get_layer(name='l4').set_weights(self.model2.get_layer(name='l4').get_weights())
        # self.target_model2.get_layer(name='l5').set_weights(self.model2.get_layer(name='l5').get_weights())
        self.target_model2.get_layer(name='l6').set_weights(self.model2.get_layer(name='l6').get_weights())
 
    def replay(self):
        if self.global_step % self.update_target_steps == 0:
            self.replace_target()
        # replay the history and train the model
        minibatch = random.sample(self.buffer, self.Batch_size)
        for state, action1, action2, reward, next_state, reward_id, done in minibatch:
            target = reward
            target1 = reward
            target2 = reward
            if not done:
                output = self.target_model.predict(next_state, verbose=0)
                k = np.max(output)
                target = (reward + self.gama * np.argmax(output))
                next_state1 = np.expand_dims(np.append(next_state[0],k), 0)
                output1 = self.target_model1.predict(next_state1, verbose=0)
                k1 = np.max(output1)
                target1 = (reward + self.gama * np.argmax(output1))
                next_state2 = np.expand_dims(np.append(next_state1[0],k1), 0)
                target2 = (reward + self.gama * np.argmax(self.target_model2.predict(next_state2, verbose=0)))
            target_f = self.model.predict(state, verbose=0)
            k = np.max(target_f)
            state1 = np.expand_dims(np.append(state[0],k),0)
            target_f1 = self.model1.predict(state1, verbose=0)
            k1 = np.max(target_f1)
            state2 = np.expand_dims(np.append(state1[0], k1), 0)
            target_f2 = self.model2.predict(state2, verbose=0)
            target_f[0][reward_id] = target
            target_f1[0][action1] = target1
            target_f2[0][action2] = target2
            self.model.fit(state, target_f, epochs=1, verbose=0)
            self.model1.fit(state1, target_f1, epochs=1, verbose=0)
            self.model2.fit(state2, target_f2, epochs=1, verbose=0)
        self.global_step += 1
 
    def Select_action(self,obs):
        # obs=np.expand_dims(obs,0)
        if random.random()<self.e_greedy:
            rt=random.choice([0,2])
            act1=random.randint(0,4)
            act2=1
        else:
            output=self.model.predict(obs, verbose=0)
            rt=np.argmax(output)
            input1 = np.expand_dims(np.append(obs[0], rt),0)
            output1 = self.model1.predict(input1, verbose=0)
            act1 = np.argmax(output1)
            input2 = np.expand_dims(np.append(input1[0], act1),0)
            output2 = self.model2.predict(input2, verbose=0)
            act2 = np.argmax(output2)
        self.e_greedy = max(
            0.01, self.e_greedy - self.e_greedy_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act1, act2, rt
 
    def _append(self, exp):
        self.buffer.append(exp)

    def main(self,J_num, M_num, O_num, J, Processing_time, D, A, Change_cutter_time, Repair_time):
        k = 0
        x=[]
        Total_tard=[]
        Total_makespan=[]
        Total_uk_ave=[]
        TR=[]
        for l in range(self.L):
            Total_reward = 0
            x.append(l+1)
            print('-----------------------开始第',l+1,'次训练------------------------------')
            obs=[0 for i in range(5)]
            obs = np.expand_dims(obs, 0)
            done=False
            Sit = Situation(J_num, M_num, O_num, J, Processing_time, D, A, Change_cutter_time, Repair_time)
            makespan_t = 0
            makespan_t1 = 0
            TR_a_t = Sit.TR_a
            TR_e_t = Sit.TR_e
            # if l == self.L - 1:
            #     curr_dir = os.getcwd()
            #     file = open(curr_dir + '\\feature\\' + str(e_ave) + str(machine) + str(job_insert) + 'result.txt', 'w+',
            #                 encoding='utf-8')
            for j in range(O_num):
                k+=1
                # print(obs)
                act1, act2, rt=self.Select_action(obs)
                # print(act1,act2)
                if act1 == 0 and act2 == 0:
                    j_i = Sit.job_rule1()
                    m_i = Sit.machine_rule1(j_i, rt)
                    at_trans = [j_i, m_i]
                if act1 == 0 and act2 == 1:
                    j_i = Sit.job_rule1()
                    m_i = Sit.machine_rule2(j_i)
                    at_trans = [j_i, m_i]
                if act1 == 0 and act2 == 2:
                    j_i = Sit.job_rule1()
                    m_i = Sit.machine_rule3(j_i)
                    at_trans = [j_i, m_i]
                if act1 == 1 and act2 == 0:
                    j_i = Sit.job_rule2()
                    m_i = Sit.machine_rule1(j_i, rt)
                    at_trans = [j_i, m_i]
                if act1 == 1 and act2 == 1:
                    j_i = Sit.job_rule2()
                    m_i = Sit.machine_rule2(j_i)
                    at_trans = [j_i, m_i]
                if act1 == 1 and act2 == 2:
                    j_i = Sit.job_rule2()
                    m_i = Sit.machine_rule3(j_i)
                    at_trans = [j_i, m_i]
                if act1 == 2 and act2 == 0:
                    j_i = Sit.job_rule3()
                    m_i = Sit.machine_rule1(j_i, rt)
                    at_trans = [j_i, m_i]
                if act1 == 2 and act2 == 1:
                    j_i = Sit.job_rule3()
                    m_i = Sit.machine_rule2(j_i)
                    at_trans = [j_i, m_i]
                if act1 == 2 and act2 == 2:
                    j_i = Sit.job_rule3()
                    m_i = Sit.machine_rule3(j_i)
                    at_trans = [j_i, m_i]
                if act1 == 3 and act2 == 0:
                    j_i = Sit.job_rule4()
                    m_i = Sit.machine_rule1(j_i, rt)
                    at_trans = [j_i, m_i]
                if act1 == 3 and act2 == 1:
                    j_i = Sit.job_rule4()
                    m_i = Sit.machine_rule2(j_i)
                    at_trans = [j_i, m_i]
                if act1 == 3 and act2 == 2:
                    j_i = Sit.job_rule4()
                    m_i = Sit.machine_rule3(j_i)
                    at_trans = [j_i, m_i]
                if act1 == 4 and act2 == 0:
                    j_i = Sit.job_rule5()
                    m_i = Sit.machine_rule1(j_i, rt)
                    at_trans = [j_i, m_i]
                if act1 == 4 and act2 == 1:
                    j_i = Sit.job_rule5()
                    m_i = Sit.machine_rule2(j_i)
                    at_trans = [j_i, m_i]
                if act1 == 4 and act2 == 2:
                    j_i = Sit.job_rule5()
                    m_i = Sit.machine_rule3(j_i)
                    at_trans = [j_i, m_i]
                # if act==0:
                #     at_trans=Sit.rule1()
                # if at==1:
                #     at_trans=Sit.rule2()
                # if at==2:
                #     at_trans=Sit.rule3()
                # if at==3:
                #     at_trans=Sit.rule4()
                # if at==4:
                #     at_trans=Sit.rule5()
                # if at==5:
                #     at_trans=Sit.rule6()
                # if at==6:
                #     at_trans=Sit.rule7()
                # at_trans=self.act[at]
                # print('这是第',j,'道工序>>', '奖励规则:', rt, '执行作业规则:',act1, ',设备规则', act2,' ','将工件',at_trans[0],'安排到机器',at_trans[1])
                Sit.scheduling(at_trans)
                obs_t=Sit.Features()
                TR_a_t1=Sit.TR_a
                TR_e_t1=Sit.TR_e
                if j==O_num-1:
                    done=True
                #obs = obs_t
                obs_t = np.expand_dims(obs_t, 0)
                # obs = np.expand_dims(obs, 0)
                # print(obs,obs_t)
                Job = Sit.Jobs
                for Ji in range(len(Job)):
                    if len(Job[Ji].End) > 0:
                        endTime = max(Job[Ji].End)
                    else:
                        endTime = 0
                    makespan_t1 = max(makespan_t1, endTime)
                if 0 == rt:
                    #r_t = Sit.reward1(obs[0][9], obs[0][7], obs_t[0][9], obs_t[0][7])
                    r_t = Sit.reward1(TR_a_t,TR_e_t,TR_a_t1,TR_e_t1)
                elif 1 == rt:
                    r_t = Sit.reward2(obs[0][0], obs_t[0][0])
                else:
                    # r_t = Sit.reward3(obs[0][7], obs_t[0][7])
                    r_t = Sit.reward3(makespan_t, makespan_t1)
                makespan_t = makespan_t1
                TR_a_t = TR_a_t1
                TR_e_t = TR_e_t1
                #--------------------
                # if l == self.L - 1:
                #     total_tadiness = 0
                #     if len(Job[Ji].End) > 0:
                #         for Ji in range(len(Job)):
                #             if max(Job[Ji].End) > D[Ji]:
                #                 total_tadiness += abs(max(Job[Ji].End) - D[Ji])
                #     T_d = total_tadiness
                #     U_k = sum(Sit.UK)/M_num
                #     M_s = makespan_t
                #     file.write(str(obs))
                #     file.write("," + str(T_d) + "," + str(U_k) + "," + str(M_s) + "," + str(rt) + "," + str(r_t))
                #     file.write("\n")
                #     file.flush()
                #     if done == True:
                #         file.close()
                #--------------------
                self._append((obs, act1, act2, r_t, obs_t, rt, done))
                if k>self.Batch_size:
                    # batch_obs, batch_action, batch_reward, batch_next_obs,done= self.sample()
                    self.replay()
                Total_reward+=r_t
                obs=obs_t
            total_tadiness=0
            makespan=makespan_t
            uk_ave=sum(Sit.UK)/M_num
            Job=Sit.Jobs
            # E=0
            # K=[i for i in range(len(Job))]
            for Ji in range(len(Job)):
                if max(Job[Ji].End)>D[Ji]:
                    total_tadiness+=abs(max(Job[Ji].End)-D[Ji])
            print('<<<<<<<<<-----------------total_tardiness:',total_tadiness,'------------------->>>>>>>>>>')
            Total_tard.append(total_tadiness)
            print('<<<<<<<<<-----------------uk_ave:', uk_ave, '------------------->>>>>>>>>>')
            Total_uk_ave.append(uk_ave)
            print('<<<<<<<<<-----------------makespan:', makespan, '------------------->>>>>>>>>>')
            Total_makespan.append(makespan)
            print('<<<<<<<<<-----------------reward:',Total_reward,'------------------->>>>>>>>>>')
            TR.append(Total_reward)
            # plt.plot(K,End,color='y')
            # plt.plot(K,D,color='r')
            # plt.show()
        # plt.plot(x,Total_tard)
        # plt.show()
        return Total_tard,Total_uk_ave,Total_makespan

    def Instance_Generator(self,M_num, E_ave, New_insert):
        '''
        :param M_num: Machine Number
        :param Initial_job: initial job number
        :param E_ave
        :return: Processing time,A:New Job arrive time,
                                    D:Deliver time,
                                    M_num: Machine Number,
                                    Op_num: Operation Number,
                                    J_num:Job NUMBER
        '''
        E_ave = E_ave
        Initial_Job_num = 5
        Op_num = [random.randint(1, 20) for i in range(New_insert + Initial_Job_num)]
        Processing_time = []
        for i in range(Initial_Job_num + New_insert):
            Job_i = []
            for j in range(Op_num[i]):
                k = random.randint(1, M_num - 2)
                T = list(range(M_num))
                random.shuffle(T)
                T = T[0:k + 1]
                O_i = list(np.ones(M_num) * (-1))
                for M_i in range(len(O_i)):
                    if M_i in T:
                        O_i[M_i] = random.randint(1, 50)
                Job_i.append(O_i)
            Processing_time.append(Job_i)
        A1 = [0 for i in range(Initial_Job_num)]
        A = np.random.exponential(E_ave, size=New_insert)
        A = [int(A[i]) for i in range(len(A))]  # New Insert Job arrive time
        A1.extend(A)
        T_ijave = []
        for i in range(Initial_Job_num + New_insert):
            Tad = []
            for j in range(Op_num[i]):
                T_ijk = [k for k in Processing_time[i][j] if k != -1]
                Tad.append(sum(T_ijk) / len(T_ijk))
            T_ijave.append(sum(Tad))
        #random.choice([0.5, 1.0, 1.5])
        D1 = [int(T_ijave[i] * random.choice([0.5, 1.0, 1.5])) for i in range(Initial_Job_num)]
        D = [int(A1[i] + T_ijave[i] * random.choice([0.5, 1.0, 1.5])) for i in range(Initial_Job_num, Initial_Job_num + New_insert)]
        D1.extend(D)
        # ?DDT=0.5
        O_num = sum(Op_num)
        J = dict(enumerate(Op_num))
        J_num = Initial_Job_num + New_insert

        # 每台设备切换时间
        Change_cutter_time = list(np.zeros(M_num))
        # 每台设备损坏后维修时间
        Repair_time = list(np.zeros(M_num))
        for i in range(M_num):
            Change_cutter_time[i] = random.randint(1, 20)
            Repair_time[i] = random.randint(1, 99)

        return Processing_time, A1, D1, M_num, Op_num, J, O_num, J_num, Change_cutter_time, Repair_time


Total_Machine=[10, 20,30]  #全部机器
Job_insert=[20,30,40]    #工件新到达个数
#Initial_job=[20,30,40]   #初始化任务数
DDT=[0.5,1.0,1.5]        #工件紧急程度
E_ave=[50,100,200]       #指数分布


def train(e_ave, machine, job_insert):
    curr_dir = os.getcwd()
    file = open(curr_dir + '\\LS\\' + str(e_ave) + str(machine) + str(job_insert)+  'result.txt', 'w+', encoding='utf-8')
    d = DQN(30, 30, 30)
    Processing_time, A, D, M_num, Op_num, J, O_num, J_num, Change_cutter_time, Repair_time = d.Instance_Generator(machine, e_ave, job_insert)
    Total_tard, Total_uk_ave, Total_makespan = d.main(J_num, M_num, O_num, J, Processing_time, D, A, Change_cutter_time, Repair_time)

    tard_ave = sum(Total_tard) / d.L
    uk_ave = sum(Total_uk_ave) / d.L
    makespan_ave = sum(Total_makespan) / d.L
    std1 = 0
    std2 = 0
    std3 = 0
    for ta in Total_tard:
        std1 += np.square(ta - tard_ave)
    for ua in Total_uk_ave:
        std2 += np.square(ua - uk_ave)
    for ma in Total_makespan:
        std3 += np.square(ma - makespan_ave)
    # 标准差
    std1 = np.sqrt(std1 / d.L)
    std2 = np.sqrt(std2 / d.L)
    std3 = np.sqrt(std3 / d.L)

    file.write(str("{:.2e}".format(tard_ave)) + "/" + str("{:.2e}".format(std1)) + "," + str(
        "{:.2e}".format(uk_ave)) + "/" + str("{:.2e}".format(std2)) + "," + str(
        "{:.2e}".format(makespan_ave)) + "/" + str("{:.2e}".format(std3)))
    file.write("\n")
    file.flush()
    file.close()

# def call_back(v):
#     print('----> callback pid:', os.getpid(),',tid:',threading.currentThread().ident,',v:',v)


if __name__ == '__main__':
    Total_Machine = [10, 20, 30]  # 全部机器
    Job_insert = [20,30,40]  # 工件新到达个数
    E_ave = [50, 100, 200]  # 指数分布

    for e_ave in E_ave:
        for machine in Total_Machine:
            for job_insert in Job_insert:
                train(e_ave, machine, job_insert)
    # pool = multiprocessing.Pool(27)
    # results = [pool.apply_async(train, args=(e_ave, machine,job_insert), callback=call_back) for e_ave in E_ave for machine in Total_Machine for job_insert in Job_insert]
    # pool.close()
    # pool.join()
