class Object:
    def __init__(self,I):
        self.I=I
        self.Start=[]
        self.End=[]
        self.T=[]
        self.assign_for=[]
 
    def _add(self,S,E,obs,t):
        #obs:���ŵĶ���
        self.Start.append(S)
        self.End.append(E)
        self.Start.sort()
        self.End.sort()
        self.T.append(t)
        self.assign_for.insert(self.End.index(E),obs)
 
    def idle_time(self):
        Idle=[]
        preJob=[]
        try:
            if self.Start[0]!=0:
                Idle.append([0,self.Start[0]])
            K=[[self.End[i],self.Start[i+1]] for i in range(len(self.End)) if self.Start[i+1]-self.End[i]>0]
            J=[[self.assign_for[i], self.assign_for[i+1]] for i in range(len(self.End)) if self.Start[i+1]-self.End[i]>0]
            Idle.extend(K)
            preJob.extend(J)
        except:
            pass
        return  Idle,preJob