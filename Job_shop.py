import numpy as np
import random
#from Instance_Generator import Processing_time,A,D,M_num,Op_num,J,O_num,J_num
from Object_for_FJSP import Object
 
class Situation:
    def __init__(self,J_num,M_num,O_num,J,Processing_time,D,Ai,Change_cutter_time, Repair_time):
        self.Ai=Ai                  #��������ʱ��
        self.D=D                    #������
        self.O_num=O_num            #��������
        self.M_num=M_num            #������
        self.J_num=J_num            #������
        self.J=J                    #������Ӧ�Ĺ�����
        self.Processing_time = Processing_time   # �ӹ�ʱ��
        self.CTK=[0 for i in range(M_num)]      #�����������һ��������깤ʱ���б�
        self.OP=[0 for i in range(J_num)]       #���������Ѽӹ��������б�
        self.UK=[0 for i in range(M_num)]       #��������ʵ��ʹ����
        self.CRJ=[0 for i in range(J_num)]      #�����깤��
        # ��������
        self.Jobs=[]
        for i in range(J_num):
            F=Object(i)
            self.Jobs.append(F)
        #������
        self.Machines = []
        for i in range(M_num):
            F = Object(i)
            self.Machines.append(F)
        #��״̬��
        self.CTJ=[0 for i in range(J_num)]      #�������һ����������ʱ��
        self.UJ=[0 for i in range(J_num)]       #��������ʱ��������
        self.TR_e=[0 for i in range(J_num)]     #��������Ԥ�ڳ�����
        self.TR_a = [0 for i in range(J_num)]  # ��������ʵ�ʳ�����

        self.Change_cutter_time=Change_cutter_time
        self.Repair_time=Repair_time
        # ---------------breakdown probability----------
        self.BP = 0.1
 
    #��������
    def _Update(self,Job,Machine):
        self.CTK[Machine]=max(self.Machines[Machine].End)
        self.OP[Job]+=1
        self.UK[Machine]=sum(self.Machines[Machine].T)/self.CTK[Machine]
        self.CRJ[Job]=self.OP[Job]/self.J[Job]
        #��״̬
        self.CTJ[Job]=max(self.Jobs[Job].End)
        if self.CTJ[Job]-self.Ai[Job] > 0:
            self.UJ[Job] = sum(self.Jobs[Job].T) / (self.CTJ[Job] - self.Ai[Job])
        T_left = 0
        for j in range(self.OP[Job] + 1, self.J[Job]):
            M_ij = []
            for k in range(self.M_num):
                if self.Processing_time[Job][j][k] > 0:
                    PT=self.Processing_time[Job][j][k]
                    M_ij.append(PT)
            T_left+=sum(M_ij)/len(M_ij)
        if self.CTJ[Job] + T_left>self.D[Job]:
            self.TR_e[Job]= (self.CTJ[Job] + T_left -self.D[Job]) / (self.CTJ[Job] + T_left - self.Ai[Job])
        if self.CTJ[Job]-self.D[Job] > 0:
            self.TR_a[Job]=(self.CTJ[Job]-self.D[Job]) / (self.CTJ[Job] - self.Ai[Job])

    #����ƽ��ʹ����
    def Features(self):
 
        #1 ����ƽ��������
        U_ave=sum(self.UK)/self.M_num
        K=0
        for uk in self.UK:
            K+=np.square(uk-U_ave)
        #2 ������ʹ���ʱ�׼��
        U_std=np.sqrt(K/self.M_num)
        #3 ƽ�����������
        CRO_ave=sum(self.OP)/self.O_num
        # #4 ƽ���������������
        # CRJ_ave=sum(self.CRJ)/self.J_num
        # K = 0
        # for uk in self.CRJ:
        #     K += np.square(uk - CRJ_ave)
        #5 ������������ʱ�׼��
        CRJ_std=np.sqrt(K/self.J_num)
        # 6 ����ʱ��ƽ��������
        UJ_ave = sum(self.UJ) / self.J_num
        # K = 0
        # for uj in self.UJ:
        #     K += np.square(uj - UJ_ave)
        # # 7 ʱ���ʹ���ʱ�׼��
        # UJ_std = np.sqrt(K / self.J_num)
        # # 8 Ԥ��ƽ���ӳ���
        # TR_e_ave = sum(self.TR_e) / self.J_num
        # K = 0
        # for uj in self.UJ:
        #     K += np.square(uj - TR_e_ave)
        # # 9 Ԥ��ƽ���ӳ��ʱ�׼��
        # TR_e_std = np.sqrt(K / self.J_num)
        # # 10 ʵ��ƽ���ӳ���
        # TR_a_ave = sum(self.TR_a) / self.J_num
        # K = 0
        # for uj in self.UJ:
        #     K += np.square(uj - TR_a_ave)
        # # 11 ʵ��ƽ���ӳ��ʱ�׼��aa
        # TR_a_std = np.sqrt(K / self.J_num)

        #6 Estimated tardiness rate Tard_e
        # T_cur=sum(self.CTK)/self.M_num
        # N_tard,N_left=0,0
        # for i in range(self.J_num):
        #     if self.J[i]>self.OP[i]:
        #         N_left+=self.J[i]-self.OP[i]
        #         T_left=0
        #         for j in range(self.OP[i]+1,self.J[i]):
        #             M_ij = []
        #             for k in range(self.M_num):
        #                 if self.Processing_time[i][j][k] > 0:
        #                     PT=self.Processing_time[i][j][k]
        #                     M_ij.append(PT)
        #             T_left+=sum(M_ij)/len(M_ij)
        #             if T_left+T_cur>self.D[i]:
        #                 N_tard+=self.J[i]-j+1
        # try:
        #     Tard_e=N_tard/N_left
        # except:
        #     Tard_e =9999
        # #7 Actual tardiness rate Tard_a
        # N_tard, N_left = 0, 0
        # for i in range(self.J_num):
        #     if self.J[i] > self.OP[i]:
        #         N_left += self.J[i] - self.OP[i]
        #         try:
        #             if self.CTK[i] > self.D[i]:
        #                 N_tard += self.J[i] - j
        #         except:
        #             pass
        # try:
        #     Tard_a = N_tard / N_left
        # except:
        #     Tard_a =9999
        return U_ave,U_std,CRO_ave,CRJ_std,UJ_ave
        #return U_ave,U_std,CRO_ave,CRJ_ave,CRJ_std,UJ_ave,UJ_std,TR_e_ave,TR_e_std,TR_a_ave,TR_a_std

    #job rule 1
    #ƽ��ʣ��ӹ�ʱ���������
    def job_rule1(self):
        # T_cur:ƽ���깤ʱ��
        T_cur = sum(self.CTK) / self.M_num
        # Tard_Job:���ܰ�����ɵĹ���
        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and self.D[i] < T_cur]
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        if Tard_Job == []:
            # ƽ��ʣ�����ʱ������job���
            Job_i = UC_Job[np.argmax([(self.D[i] - T_cur) / (self.J[i] - self.OP[i]) for i in UC_Job])]
        else:
            T_ijave = []
            for i in Tard_Job:
                Tad = []
                for j in range(self.OP[i], self.J[i]):
                    T_ijk = []
                    for k in range(self.M_num):
                        if self.Processing_time[i][j][k] != -1:
                            PT = self.Processing_time[i][j][k]
                            T_ijk.append(PT)
                    # �ӳ���ҵi��ƽ��ʣ�๤����ʱ��
                    Tad.append(sum(T_ijk) / len(T_ijk))
                T_ijave.append((T_cur + sum(Tad) - self.D[i]))
            Job_i = Tard_Job[np.argmax(T_ijave)]  # ƽ��ʣ�����ʱ������job���
        return Job_i

    # job rule 2
    # ��Сʱ������������
    def job_rule2(self):
        # T_cur:ƽ���깤ʱ��
        T_cur = sum(self.CTK) / self.M_num
        # Tard_Job:���ܰ�����ɵĹ���
        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and self.D[i] < T_cur]
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        if Tard_Job == []:
            Job_i = UC_Job[np.argmin([self.UJ[i] * (self.D[i] - T_cur) for i in UC_Job])]
        else:
            T_ijave = []
            for i in Tard_Job:
                Tad = []
                for j in range(self.OP[i], self.J[i]):
                    T_ijk = []
                    for k in range(self.M_num):
                        if self.Processing_time[i][j][k] != -1:
                            PT = self.Processing_time[i][j][k]
                            T_ijk.append(PT)
                    Tad.append(sum(T_ijk) / len(T_ijk))
                T_ijave.append((T_cur + sum(Tad) - self.D[i]) / (self.UJ[i] + 1))
            Job_i = Tard_Job[np.argmax(T_ijave)]
        return Job_i

    # job rule 3
    # ��С�깤����ҵ����
    def job_rule3(self):
        # T_cur:ƽ���깤ʱ��
        T_cur = sum(self.CTK) / self.M_num
        # Tard_Job:���ܰ�����ɵĹ���
        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and self.D[i] < T_cur]
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        if Tard_Job == []:
            Job_i = UC_Job[np.argmin([self.CRJ[i] * (self.D[i] - T_cur) for i in UC_Job])]
        else:
            T_ijave = []
            for i in Tard_Job:
                Tad = []
                for j in range(self.OP[i], self.J[i]):
                    T_ijk = []
                    for k in range(self.M_num):
                        if self.Processing_time[i][j][k] != -1:
                            PT = self.Processing_time[i][j][k]
                            T_ijk.append(PT)
                    Tad.append(sum(T_ijk) / len(T_ijk))
                T_ijave.append((T_cur + sum(Tad) - self.D[i]) / (self.CRJ[i] + 1))
            Job_i = Tard_Job[np.argmax(T_ijave)]
        return Job_i

    # job rule 4
    # �ٽ�������Ԥ��ʣ��ӹ�ʱ������ҵ����
    def job_rule4(self):
        # T_cur:ƽ���깤ʱ��
        T_cur = sum(self.CTK) / self.M_num
        # Tard_Job:���ܰ�����ɵĹ���
        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and self.D[i] < T_cur]
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        T_ijave = []
        for i in range(self.J_num):
            Tad = []
            for j in range(self.OP[i], self.J[i]):
                T_ijk = []
                for k in range(self.M_num):
                    if self.Processing_time[i][j][k] != -1:
                        PT = self.Processing_time[i][j][k]
                        T_ijk.append(PT)
                Tad.append(sum(T_ijk) / len(T_ijk))
            T_ijave.append(sum(Tad))
        if Tard_Job==[]:
            Job_i=UC_Job[np.argmin([(self.D[i]-T_cur)/T_ijave[i] for i in UC_Job])]
        else:
            Job_i=Tard_Job[np.argmax([(T_cur+T_ijave[i]-self.D[i]) for i in Tard_Job])]
        return Job_i

    # job rule 5
    # �������������
    def job_rule5(self):
        # T_cur:ƽ���깤ʱ��
        T_cur = sum(self.CTK) / self.M_num
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        # if rt == 1:
        #     Job_i = random.choice(UC_Job)
        #     return Job_i
        T_ijave = []
        for i in UC_Job:
            Tad = []
            for j in range(self.OP[i], self.J[i]):
                T_ijk = []
                for k in range(self.M_num):
                    if self.Processing_time[i][j][k] != -1:
                        PT = self.Processing_time[i][j][k]
                        T_ijk.append(PT)
                Tad.append(sum(T_ijk) / len(T_ijk))
            T_ijave.append(T_cur + sum(Tad) - self.D[i])
        Job_i = UC_Job[np.argmax(T_ijave)]
        return Job_i

    # machine rule 1
    # ������С�Ҽӹ����ʱ���ٵ��豸�����ױ�ѡ��
    def machine_rule1(self, Job_i, rt):
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]  # ����i��arrival time
        A_ij = self.Ai[Job_i]  # ����i��arrival time
        On = len(self.Jobs[Job_i].End)
        Mk = []
        Mk1 = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                PT = self.Processing_time[Job_i][On][i]
                # �����л�ʱ��
                if self.change_cutter(Job_i, i) == 1:
                    PT += self.Change_cutter_time[i]
                Mk.append(max(C_ij + PT, A_ij, self.CTK[i]))
                Mk1.append(max(C_ij+PT, A_ij, self.CTK[i])*self.UK[i])
            else:
                Mk.append(999999)
                Mk1.append(999999)
        # print('This is from rule 1:',Mk)
        if rt == 1:
            Machine = np.argmin(Mk1)
        else:
            Machine = np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Machine

    # machine rule 2
    # ��С���ʱ������
    def machine_rule2(self, Job_i):
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]  # ����i��arrival time
        A_ij = self.Ai[Job_i]  # ����i��arrival time
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                PT = self.Processing_time[Job_i][On][i]
                # �����л�ʱ��
                if self.change_cutter(Job_i, i) == 1:
                    PT += self.Change_cutter_time[i]
                Mk.append(max(C_ij+PT, A_ij, self.CTK[i]))
            else:
                Mk.append(999999)
        # print('This is from rule 1:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Machine

    # machine rule 3
    # �豸����������
    def machine_rule3(self, Job_i):
        On = len(self.Jobs[Job_i].End)
        if random.random()<0.5:
            U=[]
            for i in range(len(self.UK)):
                if self.Processing_time[Job_i][On][i]==-1:
                    U.append(9999)
                else:
                    U.append(self.UK[i])
            Machine=np.argmin(U)
        else:
            MT=[]
            for j in range(self.M_num):
                if self.Processing_time[Job_i][On][j]==-1:
                    MT.append(9999)
                else:
                    MT.append(sum(self.Machines[j].T))
            Machine=np.argmin(MT)
        # print('This is from rule 1:',Machine)
        return Machine

    #Composite dispatching rule 1
    #return Job,Machine
    def rule1(self):
        #T_cur:ƽ���깤ʱ��
        T_cur = sum(self.CTK) / self.M_num
        #Tard_Job:���ܰ�����ɵĹ���
        Tard_Job=[i for i in range(self.J_num) if self.OP[i]<self.J[i] and self.D[i]<T_cur]
        UC_Job=[j for j in range(self.J_num) if self.OP[j]<self.J[j]]
        if Tard_Job==[]:
            #ƽ��ʣ�����ʱ������job���
            Job_i=UC_Job[np.argmax([(self.D[i]-T_cur)/(self.J[i]-self.OP[i]) for i in UC_Job ])]
        else:
            T_ijave=[]
            for i in Tard_Job:
                Tad=[]
                for j in range(self.OP[i],self.J[i]):
                    T_ijk = []
                    for k in range(self.M_num):
                        if self.Processing_time[i][j][k] != -1:
                            PT = self.Processing_time[i][j][k]
                            T_ijk.append(PT)
                    #�ӳ���ҵi��ƽ��ʣ�๤����ʱ��
                    Tad.append(sum(T_ijk)/len(T_ijk))
                T_ijave.append(T_cur+sum(Tad)-self.D[i])
            Job_i=Tard_Job[np.argmax(T_ijave)]#ƽ��ʣ�����ʱ������job���
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij =self.Ai[Job_i]  # ����i��arrival time
        A_ij = self.Ai[Job_i]  # ����i��arrival time
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                PT = self.Processing_time[Job_i][On][i]
                # �����л�ʱ��
                if self.change_cutter(Job_i, i) == 1:
                    PT += self.Change_cutter_time[i]
                Mk.append(max(C_ij+PT, A_ij, self.CTK[i]))
            else:
                Mk.append(9999)
        # print('This is from rule 1:',Mk)
        Machine=np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i,Machine
 
    #Composite dispatching rule 2
    #return Job,Machine
    def rule2(self):
        # T_cur:ƽ���깤ʱ��
        T_cur = sum(self.CTK) / self.M_num
        # Tard_Job:���ܰ�����ɵĹ���
        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and self.D[i] < T_cur]
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        T_ijave = []
        for i in range(self.J_num):
            Tad = []
            for j in range(self.OP[i], self.J[i]):
                T_ijk = []
                for k in range(self.M_num):
                    if self.Processing_time[i][j][k] != -1:
                        PT = self.Processing_time[i][j][k]
                        T_ijk.append(PT)
                Tad.append(sum(T_ijk) / len(T_ijk))
            T_ijave.append(sum(Tad))
        if Tard_Job==[]:
            Job_i=UC_Job[np.argmin([(self.D[i]-T_cur)/T_ijave[i] for i in UC_Job])]
        else:
            Job_i=Tard_Job[np.argmax([T_cur+T_ijave[i]-self.D[i] for i in Tard_Job ])]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij =self.Ai[Job_i]  # ����i��arrival time
        A_ij = self.Ai[Job_i]  # ����i��arrival time
        # print(A_ij)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                PT = self.Processing_time[Job_i][On][i]
                # �����л�ʱ��
                if self.change_cutter(Job_i, i) == 1:
                    PT += self.Change_cutter_time[i]
                Mk.append(max(C_ij+PT, A_ij, self.CTK[i]))
            else:
                Mk.append(9999)
        # print('This is from rule 2:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 2:',Machine)
        return Job_i,Machine
 
    # #Composite dispatching rule 3
    # def rule3(self):
    #     # T_cur:ƽ���깤ʱ��
    #     T_cur = sum(self.CTK) / self.M_num
    #     # Tard_Job:���ܰ�����ɵĹ���
    #     UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
    #     T_ijave = []
    #     for i in UC_Job:
    #         Tad = []
    #         for j in range(self.OP[i], self.J[i]):
    #             T_ijk = []
    #             for k in range(self.M_num):
    #                 if self.Processing_time[i][j][k] != -1:
    #                     PT = self.Processing_time[i][j][k]
    #                     T_ijk.append(PT)
    #             Tad.append(sum(T_ijk) / len(T_ijk))
    #         T_ijave.append(T_cur + sum(Tad) - self.D[i])
    #     Job_i = UC_Job[np.argmax(T_ijave)]
    #     On = len(self.Jobs[Job_i].End)
    #     if random.random()<0.5:
    #         U=[]
    #         for i in range(len(self.UK)):
    #             if self.Processing_time[Job_i][On][i]==-1:
    #                 U.append(9999)
    #             else:
    #                 U.append(self.UK[i])
    #         Machine=np.argmin(U)
    #     else:
    #         MT=[]
    #         for j in range(self.M_num):
    #             if self.Processing_time[Job_i][On][j]==-1:
    #                 MT.append(9999)
    #             else:
    #                 MT.append(sum(self.Machines[j].T))
    #         Machine=np.argmin(MT)
    #     # print('This is from rule 3:',Machine)
    #     return Job_i,Machine

    # Composite dispatching rule 3
    def rule3(self):
        # T_cur:ƽ���깤ʱ��
        T_cur = sum(self.CTK) / self.M_num
        # Tard_Job:���ܰ�����ɵĹ���
        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and self.D[i] < T_cur]
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        if Tard_Job == []:
            Job_i = UC_Job[np.argmin([self.UJ[i] * (self.D[i] - T_cur) for i in UC_Job])]
        else:
            T_ijave = []
            for i in Tard_Job:
                Tad = []
                for j in range(self.OP[i], self.J[i]):
                    T_ijk = []
                    for k in range(self.M_num):
                        if self.Processing_time[i][j][k] != -1:
                            PT = self.Processing_time[i][j][k]
                            T_ijk.append(PT)
                    Tad.append(sum(T_ijk) / len(T_ijk))
                T_ijave.append(1 / (self.UJ[i] + 1) * (T_cur + sum(Tad) - self.D[i]))
            Job_i = Tard_Job[np.argmax(T_ijave)]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.Ai[Job_i]  # ����i��arrival time
        A_ij = self.Ai[Job_i]  # ����i��arrival time
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                PT = self.Processing_time[Job_i][On][i]
                # �����л�ʱ��
                if self.change_cutter(Job_i, i) == 1:
                    PT += self.Change_cutter_time[i]
                Mk.append(max(C_ij + PT, A_ij, self.CTK[i]))
            else:
                Mk.append(9999)
        # print('This is from rule 7:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 7:',Machine)
        return Job_i, Machine

    #Composite dispatching rule 4
    def rule4(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        Job_i= random.choice(UC_Job)
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij =self.Ai[Job_i]  # ����i��arrival time
        A_ij = self.Ai[Job_i]  # ����i��arrival time
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                PT = self.Processing_time[Job_i][On][i]
                # �����л�ʱ��
                if self.change_cutter(Job_i, i) == 1:
                    PT += self.Change_cutter_time[i]
                Mk.append(max(C_ij+PT, A_ij, self.CTK[i]))
            else:
                Mk.append(9999)
        # print('This is from rule 4:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 4:',Machine)
        return Job_i,Machine
 
    #Composite dispatching rule 5
    def rule5(self):
        # T_cur:ƽ���깤ʱ��
        T_cur = sum(self.CTK) / self.M_num
        # Tard_Job:���ܰ�����ɵĹ���
        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and self.D[i] < T_cur]
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        if Tard_Job==[]:
            Job_i=UC_Job[np.argmin([self.CRJ[i]*(self.D[i]-T_cur) for i in UC_Job])]
        else:
            T_ijave = []
            for i in Tard_Job:
                Tad = []
                for j in range(self.OP[i], self.J[i]):
                    T_ijk = []
                    for k in range(self.M_num):
                        if self.Processing_time[i][j][k] != -1:
                            PT = self.Processing_time[i][j][k]
                            T_ijk.append(PT)
                    Tad.append(sum(T_ijk) / len(T_ijk))
                T_ijave.append(1/(self.CRJ[i]+1)*(T_cur + sum(Tad) - self.D[i]))
            Job_i = Tard_Job[np.argmax(T_ijave)]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij =self.Ai[Job_i]  # ����i��arrival time
        A_ij = self.Ai[Job_i]  # ����i��arrival time
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                PT = self.Processing_time[Job_i][On][i]
                # �����л�ʱ��
                if self.change_cutter(Job_i, i) == 1:
                    PT += self.Change_cutter_time[i]
                Mk.append(max(C_ij+PT, A_ij, self.CTK[i]))
            else:
                Mk.append(9999)
        # print('This is from rule 5:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 5:',Machine)
        return Job_i, Machine
 
    #Composite dispatching rule 6
    #return Job,Machine
    def rule6(self):
        # T_cur:ƽ���깤ʱ��
        T_cur = sum(self.CTK) / self.M_num
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        T_ijave = []
        for i in UC_Job:
            Tad = []
            for j in range(self.OP[i], self.J[i]):
                T_ijk = []
                for k in range(self.M_num):
                    if self.Processing_time[i][j][k] != -1:
                        PT = self.Processing_time[i][j][k]
                        T_ijk.append(PT)
                Tad.append(sum(T_ijk) / len(T_ijk))
            T_ijave.append(T_cur + sum(Tad) - self.D[i])
        Job_i = UC_Job[np.argmax(T_ijave)]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij =self.Ai[Job_i]  # ����i��arrival time
        A_ij = self.Ai[Job_i]  # ����i��arrival time
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                PT = self.Processing_time[Job_i][On][i]
                # �����л�ʱ��
                if self.change_cutter(Job_i,i) == 1:
                    PT += self.Change_cutter_time[i]
                Mk.append(max(C_ij+PT, A_ij, self.CTK[i]))
            else:
                Mk.append(9999)
        Machine = np.argmin(Mk)
        # print('this is from rule 6:',Mk)
        # print('This is from rule 6:',Machine)
        return Job_i,Machine

    # single rule1
    def fifo(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        T_ijave = []
        for i in UC_Job:
            T_ijave.append(self.Ai[i])
        Job_i = UC_Job[np.argmin(T_ijave)]
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(i)
        Machine = random.choice(Mk)
        return Job_i, Machine

    # single rule2
    def lifo(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        T_ijave = []
        for i in UC_Job:
            T_ijave.append(self.D[i])
        Job_i = UC_Job[np.argmax(T_ijave)]
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(i)
        Machine = random.choice(Mk)
        return Job_i, Machine

    # single rule3
    def edd(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        T_ijave = []
        for i in UC_Job:
            T_ijave.append(self.D[i])
        Job_i = UC_Job[np.argmin(T_ijave)]
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(i)
        Machine = random.choice(Mk)
        return Job_i, Machine

    # single rule4
    def mrt(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        T_ijave = []
        for i in UC_Job:
            try:
                C_ij = max(self.Jobs[i].End)
            except:
                C_ij = self.Ai[i]  # ����i��arrival time
            T_ijave.append(self.D[i] - C_ij)
        Job_i = UC_Job[np.argmax(T_ijave)]
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(i)
        Machine = random.choice(Mk)
        return Job_i, Machine

    # single rule5
    def spt(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        T_ijave = []
        for i in UC_Job:
            j = len(self.Jobs[i].End)
            PT = 0
            m_n = 0
            for k in range(len(self.CTK)):
                if self.Processing_time[i][j][k] != -1:
                    m_n += 1
                    PT = self.Processing_time[i][j][k]
                    # �����л�ʱ��
                    # if self.change_cutter(i, k) == 1:
                    #     PT += self.Change_cutter_time[k]
            T_ijave.append(PT/m_n)
        Job_i = UC_Job[np.argmin(T_ijave)]
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(i)
        Machine = random.choice(Mk)
        return Job_i, Machine

    # single rule6
    def lpt(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        T_ijave = []
        for i in UC_Job:
            j = len(self.Jobs[i].End)
            PT = 0
            m_n = 0
            for k in range(len(self.CTK)):
                if self.Processing_time[i][j][k] != -1:
                    m_n += 1
                    PT = self.Processing_time[i][j][k]
                    # �����л�ʱ��
                    # if self.change_cutter(i, k) == 1:
                    #     PT += self.Change_cutter_time[k]
            T_ijave.append(PT / m_n)
        Job_i = UC_Job[np.argmax(T_ijave)]
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(i)
        Machine = random.choice(Mk)
        return Job_i, Machine

    # single rule7
    def lor(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        T_ijave = []
        for i in UC_Job:
            T_ijave.append(self.OP[i])
        Job_i = UC_Job[np.argmin(T_ijave)]
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(i)
        Machine = random.choice(Mk)
        return Job_i, Machine

    # single rule8
    def mor(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        T_ijave = []
        for i in UC_Job:
            T_ijave.append(self.J[i] - self.OP[i])
        Job_i = UC_Job[np.argmax(T_ijave)]
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(i)
        Machine = random.choice(Mk)
        return Job_i, Machine

    # single rule9
    def stpt(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        T_ijave = []
        for i in UC_Job:
            T_ijave.append(self.D[i] - self.Ai[i])
        Job_i = UC_Job[np.argmin(T_ijave)]
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(i)
        Machine = random.choice(Mk)
        return Job_i, Machine

    # single rule10
    def ltpt(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        T_ijave = []
        for i in UC_Job:
            T_ijave.append(self.D[i] - self.Ai[i])
        Job_i = UC_Job[np.argmax(T_ijave)]
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(i)
        Machine = random.choice(Mk)
        return Job_i, Machine

    def randomjob(self):
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        Job_i = random.choice(UC_Job)
        On = len(self.Jobs[Job_i].End)
        Mk = []
        for i in range(len(self.CTK)):
            if self.Processing_time[Job_i][On][i] != -1:
                Mk.append(i)
        Machine = random.choice(Mk)
        return Job_i, Machine

    def scheduling(self,action):
        Job,Machine=action[0],action[1]
        O_n=len(self.Jobs[Job].End)
        # print(Job, Machine,O_n)
        Idle,preSubJob=self.Machines[Machine].idle_time()
        try:
            last_ot=max(self.Jobs[Job].End)         #�ϵ�����ӹ�ʱ��
        except:
            last_ot=0
        try:
            last_mt=max(self.Machines[Machine].End) #��������깤ʱ��
        except:
            last_mt=0
        Start_time=max(last_ot,last_mt)
        PT = self.Processing_time[Job][O_n][Machine]  # ����ӹ�ʱ��
        # �豸ͻ�����ϸ���
        break_down = random.random()
        uk_bp = np.percentile(self.UK, 90)
        # �豸��������ǰ10%���豸�������׳��ֹ���
        if self.UK[Machine] >= uk_bp:
            break_down = min(break_down, random.random())
        if break_down < self.BP:
            PT += self.Repair_time[Machine]
        is_LS = 0
        for i in range(len(Idle)):
            pt=PT
            if Idle[i][1] - Idle[i][0] > pt:
                if len(preSubJob) > 0 and preSubJob[i][0] == preSubJob[i][1]:
                    if preSubJob[i][0] != Job:
                        pt += 2*self.Change_cutter_time[Machine]
                else:
                    if len(preSubJob) > 0 and preSubJob[i][0] != Job and preSubJob[i][1] != Job:
                        pt += self.Change_cutter_time[Machine]
            if Idle[i][1]-Idle[i][0]>pt:
                if Idle[i][0]>last_ot:
                    is_LS = 1
                    PT=pt
                    Start_time=Idle[i][0]
                    pass
                if Idle[i][0]<last_ot and Idle[i][1]-last_ot>pt:
                    is_LS = 1
                    PT=pt
                    Start_time=last_ot
                    pass
        if is_LS == 0 and self.change_cutter(Job,Machine) == 1:
            PT += self.Change_cutter_time[Machine]
        end_time=Start_time+PT
        self.Machines[Machine]._add(Start_time,end_time,Job,PT)
        self.Jobs[Job]._add(Start_time,end_time,Machine,PT)
        self._Update(Job,Machine)

    # def reward1(self,Ta_t,Te_t,Ta_t1,Te_t1,U_t,U_t1):
    #     '''
    #            :param Ta_t: Tard_a(t)
    #            :param Te_t: Tard_e(t)
    #            :param Ta_t1: Tard_a(t+1)
    #            :param Te_t1: Tard_e(t+1)
    #            :param U_t: U_ave(t)
    #            :param U_t1: U_ave(t+1)
    #            :return: reward
    #     '''
    #     if Ta_t1<Ta_t:
    #        rt=1
    #     else:
    #         if Ta_t1>Ta_t:
    #             rt=-1
    #         else:
    #             if Te_t1<Te_t:
    #                 rt=1
    #             else:
    #                 if Te_t1>Te_t:
    #                     rt=-1
    #                 else:
    #                     if U_t1>U_t:
    #                         rt=1
    #                     else:
    #                         if U_t1>0.95*U_t:
    #                             rt=0
    #                         else:
    #                             rt=-1
    #     return rt

    def reward1(self,Ta_t,Te_t,Ta_t1,Te_t1):
        '''
               :param Ta_t: Tard_a(t)
               :param Te_t: Tard_e(t)
               :param Ta_t1: Tard_a(t+1)
               :param Te_t1: Tard_e(t+1)
               :return: reward
        '''
        if Ta_t1<Ta_t:
           rt=1
        else:
            if Ta_t1>Ta_t:
                rt=-1
            else:
                if Te_t1<Te_t:
                    rt=1
                else:
                    if Te_t1>Te_t:
                        rt=-1
                    else:
                        rt = 0
        return rt

    # def reward2(self,Ta_t,Te_t,Ta_t1,Te_t1,UJ_t,UJ_t1):
    #     '''
    #            :param Ta_t: Tard_a(t)
    #            :param Te_t: Tard_e(t)
    #            :param Ta_t1: Tard_a(t+1)
    #            :param Te_t1: Tard_e(t+1)
    #            :param UJ_t: UJ_ave(t)
    #            :param UJ_t1: UJ_ave(t+1)
    #            :return: reward
    #     '''
    #     if Ta_t1<Ta_t:
    #        rt=1
    #     else:
    #         if Ta_t1>Ta_t:
    #             rt=-1
    #         else:
    #             if Te_t1<Te_t:
    #                 rt=1
    #             else:
    #                 if Te_t1>Te_t:
    #                     rt=-1
    #                 else:
    #                     if UJ_t1>UJ_t:
    #                         rt=1
    #                     else:
    #                         if UJ_t1>0.95*UJ_t:
    #                             rt=0
    #                         else:
    #                             rt=-1
    #     return rt

    def reward2(self,U_t,U_t1):
        '''
               :param U_t: U_ave(t)
               :param U_t1: U_ave(t+1)
               :return: reward
        '''
        if U_t1 > U_t:
            rt = 1
        else:
            if U_t1 > 0.95 * U_t:
                rt = 0
            else:
                rt = -1
        return rt

    def reward3(self,MS_t,MS_t1):
        '''
               :param MS_t: makespan(t)
               :param MS_t1: makespan(t+1)
               :return: reward
        '''
        if MS_t1 <= MS_t:
            rt = 1
        else:
            if MS_t1 < 1.05 * MS_t or MS_t == 0:
                rt = 0
            else:
                rt = -1
        return rt

    # def reward3(self,UJ_t,UJ_t1):
    #     '''
    #            :param UJ_t: UJ_ave(t)
    #            :param UJ_t1: UJ_ave(t+1)
    #            :return: reward
    #     '''
    #     if UJ_t1 > UJ_t:
    #         rt = 1
    #     else:
    #         if UJ_t1 > 0.95 * UJ_t:
    #             rt = 0
    #         else:
    #             rt = -1
    #     return rt



    # ��ǰ�豸��һ�μӹ�����ҵ�뵱ǰ��ͬ�������л�ʱ��
    def change_cutter(self,Job,Machine):
        assigned_jobs = self.Machines[Machine].assign_for
        if len(assigned_jobs) != 0 and assigned_jobs[-1] != Job:
            return 1
        return 0
#Sit=Situation(J_num,M_num,O_num,J,Processing_time,D,A)