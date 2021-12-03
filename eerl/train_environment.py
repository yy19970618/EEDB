import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import database
from database import DatabaseOp
import copy
from agent import Agent
from heap import SlideBar
class Environment:
    old_state_input=[[]]
    state_input=[[]]
    action_num = 63
    delta = [0,8,24,28,37,46,51,54]
    index = []
    max_cost=-100
    model=None
    replay_memory_store=deque()
    memory_size = 5000
    memory_counter = 0
    BATCH = 5
    step_index=0
    gamma=0.9
    dbop=None
    exp_store=[]
    exp_flag = 0
    exp_ind=0
    beta=0.8
    loss_store=[]
    cost_store=[]
    query_store=[]
    heap=None
    rate=None
    reward=[]
    key=0.4
    key2=0.8
    gamma=0.8
    my_model=None
    f1=open('input.txt','w')
    f2=open('output.txt','w')
    fr = open('reward.txt', 'w+')
    max_dele=0
    max_update=0
    max_none=0
    random.seed(6)
    np.set_printoptions(threshold=np.inf)
    select_col=[[[13, 14, 15, 17, 18, 19]],
				[[1, 7], [9, 14, 16], [33, 36]],
				[[9, 20, 21], [29, 33, 34]],
				[[1], [11, 14, 15], [25, 26, 27], [30, 33], [48], [52, 53], [58]],
				[[13, 14, 15, 19]],
				[[1, 4], [11, 15, 19], [26], [30], [55, 58]],
				[[1, 4], [9, 10, 11, 14, 15], [25, 26, 27], [29, 30, 33], [38, 42], [52, 53], [55, 58]],
				[[9, 10, 11, 13, 14, 15], [25, 26], [29, 33], [38, 39], [47, 48, 50], [58]],
				[[1, 2, 3, 4, 5, 6], [8, 9, 14, 15, 17], [26], [29, 30, 33]],
				[[25, 26], [47, 49, 50], [55, 58]],
				[[9, 19, 20, 21, 23], [28, 29]],
				[[1], [29, 30], [37]],
				[[10, 14, 15, 19], [38, 42]],
				[[38, 41, 42, 43], [47, 48, 51]],
				[[1, 2], [9, 13], [29, 30, 32, 33]],
				[[10, 13, 14, 15, 22, 23], [38, 41, 43, 44]],
				[[9, 11, 20, 21], [25, 26], [29, 31], [48], [56, 58]],
				[[1, 5, 6], [30]]]
    def __init__(self):
        a=Agent()
        self.model=a.get_model(63,30)
        self.model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())
        self.model.summary()
        self.model.optimizer = tf.optimizers.Adam(learning_rate=0.1)
        self.model.loss_func = tf.losses.MeanSquaredError()

#        self.my_model=tf.keras.models.load_model('my_model.h5')
        
        i_state=[0]*61
        self.state_input[0].append(i_state)
        self.heap=SlideBar()
        for i in range(18):
            self.query_store.append([])
        for i in range(100):
            a=np.random.randint(0, 18)
            self.exp_store.append(a)
            self.heap.InsertItem(self.select_col[a])
        self.dbop=DatabaseOp(self.exp_store)
    def select_action(self,s_in):
#        res=self.my_model.predict(s_in)
#        res=np.array(res)
#        r=np.unravel_index(np.argmax(res[0]),res[0].shape)
#       return r[0],r[1]
        if np.random.uniform() < 0.1:
            a=np.random.randint(0, len(self.state_input[0]))
            b=0
            if a!=0:
                b=np.random.randint(0, 63)
            else:
                b = np.random.randint(0, 62)
            b=np.random.randint(0, 63)
            return [],a,b
        if np.random.uniform()<self.beta:
            a, b = self.exp_method()
            return [], a, b
        else:
            res=self.model.predict(s_in)
            res=np.array(res)
            r=np.unravel_index(np.argmax(res[0]),res[0].shape)
            return res,r[0],r[1]
    def exp_method(self):
        g=np.random.uniform()
        if g<0.4:
            a=np.random.randint(0, 100)
            s=[]
            for i in range(61):
                if self.rate[a][i]!=0:
                    s.append(i)
            d=[0]*61
            e=d.copy()
            while len(s)!=0:
                c=np.random.randint(0, len(s))
                d[s[c]]=1
                if d not in self.index:
                    return self.state_input[0].index(e),s[c]
                else:
                    del s[c]
                    e=d.copy()
            return self.state_input[0].index(e),0
        elif g>=0.4 and g<0.8:
            pir=[37,46,54,47,57,24,26,51,52,
	            6,0,29,8,28,32,18,
	            19,20,3,18,12,38,22,40,41,42,43,21
	            ]
            if len(self.index)==0:
                return 0,0
            a=np.random.randint(1,len(self.state_input[0]))
            b=self.state_input[0][a]
            s=[]
            for i in range(61):
                if b[i]==1 and i in pir:
                    s.append(i)
            for i in range(61):
                if b[i]==1 and i not in pir:
                    s.append(i)
            for i in range(len(s)-1):
                s2=[0]*61
                for j in range(i):
                    s2[s[j]]=1
                if i!=0 and s2 in self.state_input[0]:
                    return self.state_input[0].index(s2),62
            return a,0
        else:
            if len(self.index)==0:
                return 0,0
            a=np.random.randint(0,len(self.index))
            b=self.index[a]
            s=[]
            for i in range(61):
                if b[i]==1:
                    s.append(i)
            c=self.state_input[0].index(b)
            if self.heap.FindRate(s)>6:
                return c,0 
            else:
                return c,62
    def findTable(self,action):
        tb=0
        if action<8:
            tb=0
        elif action<24:
            tb=1
        elif action<28:
            tb=2
        elif action<37:
            tb=3
        elif action<46:
            tb=4
        elif action<51:
            tb=5
        elif action<54:
            tb=6
        else:
            tb=7
        return tb
    def exp_method2(self):
        t=[0,8,24,28,37,46,51,54,61]
        s=len(self.state_input[0])
        c=np.random.randint(0, s)
        if c==0:
            return c,np.random.randint(1, 61)
        if s<50:
            k=0
            for i in range(61):
                if self.state_input[0][c][i]==1:
                    k=self.findTable(i)
                    break
            l=np.random.randint(t[k], t[k+1])
            return c,l
        else:
            return c,62
      
    def step(self,res,s_in,action_index,action):
        old_state=copy.deepcopy(self.state_input)
        error_f=0
        if action_index==0:
            if action>0 and action<self.action_num-1:
                c=[0]*61
                c[action-1]=1
                if c not in self.index:
                    state_in=copy.deepcopy(self.state_input[0][action_index])
                    if action-1==61:
                        print("00")
                    self.dbop.addIndex(state_in,action-1,1)
                    self.index.append(c)
                    self.state_input[0].append(c)
                    self.state_input[0].sort()
                else:
                    print("err")
                    error_f=1
            else:
                action=np.random.randint(1, 62)
                c=[0]*61
                c[action-1]=1
                if c not in self.index:
                    state_in=copy.deepcopy(self.state_input[0][action_index])
                    self.dbop.addIndex(state_in,action-1,1)
                    self.index.append(c)
                    self.state_input[0].append(c)
                    self.state_input[0].sort()
                else:
                    print("err")
                    error_f=1

        elif action==0:
            old_state=[]
            old_state.append(copy.deepcopy(self.state_input[0]))
        elif action==self.action_num-1:
            print("del")
            self.dbop.dropIndex(copy.deepcopy(self.state_input[0][action_index]))
            self.index.remove(copy.deepcopy(self.state_input[0][action_index]))
            print(id(self.state_input)==id(old_state))
            del self.state_input[0][action_index]
        else:
            c=copy.deepcopy(self.state_input[0][action_index])
            c[action-1]=1
            if c not in self.index and self.isLegal(action_index,action-1):
                print("add")
                state_in=copy.deepcopy(self.state_input[0][action_index])
                if action==61:
                        print("00")
                self.dbop.addIndex(state_in,action-1,0)
                self.state_input[0][action_index][action-1]=1
                self.state_input[0].sort()
            else:
                print("err")
                error_f=1
        dele=self.exp_store[self.exp_ind]
        cost,select,ccc= self.dbop.getCost(dele,self.step_index)
        self.fr.write(str(cost)+'\n')
        self.reward.append(cost)
        for i in range(18):
            self.query_store[i].append(ccc[i])
        if self.step_index%10==0:
            self.cost_store.append(cost)
        self.heap.DeleteItem(self.select_col[self.exp_store[self.exp_ind]])
        self.exp_store[self.exp_ind]=select
        self.heap.InsertItem(self.select_col[select])
        self.exp_ind+=1
        if self.exp_ind==100:
            self.exp_ind=0
        if error_f==1:
            cost=self.max_cost
        return old_state, cost
    def save_store(self, current_state, current_action_index,current_action,current_cost,next_state):
        
        self.replay_memory_store.append((
            current_state,
            current_action_index,
            current_action,
            current_cost,
            next_state
            ))
        if len(self.replay_memory_store) > self.memory_size:
            self.replay_memory_store.popleft()
        self.memory_counter += 1

    def experience_replay(self):
        batch = self.BATCH if self.memory_counter > self.BATCH else self.memory_counter
        minibatch = random.sample(self.replay_memory_store, batch)
        batch_state = None
        batch_action_index = None
        batch_action = None
        batch_cost = None
        batch_next_state = None
        maxl=0
        maxl2=0
        for index in range(len(minibatch)):
            if batch_state is None:
                batch_state = copy.deepcopy(minibatch[index][0])
                maxl=len(minibatch[index][0][0])
            elif batch_state is not None:
                batch_state.append(minibatch[index][0][0])
                if maxl<len(minibatch[index][0][0]):
                    maxl=len(minibatch[index][0][0])
            if batch_action_index is None:
                batch_action_index = minibatch[index][1]
            elif batch_action_index is not None:
                batch_action_index = np.append(batch_action_index, minibatch[index][1])
            if batch_action  is None:
                batch_action = minibatch[index][2]
            elif batch_action is not None:
                batch_action = np.append(batch_action, minibatch[index][2])
            if batch_cost is None:
                batch_cost = minibatch[index][3]
            elif batch_cost is not None:
                batch_cost = np.append(batch_cost, minibatch[index][3])
            if batch_next_state is None:
                batch_next_state = copy.deepcopy(minibatch[index][4])
                maxl2=len(minibatch[index][4][0])
            elif batch_next_state is not None:
                batch_next_state.append(minibatch[index][4][0])
                if maxl2<len(minibatch[index][4][0]):
                    maxl2=len(minibatch[index][4][0])
        for i in range(self.BATCH):
            if len(batch_state[i])<maxl:
                for j in range(len(batch_state[i]),maxl):
                    batch_state[i].append([-1.]*61)
        for i in range(self.BATCH):
            if len(batch_next_state[i])<maxl2:
                for j in range(len(batch_next_state[i]),maxl2):
                    batch_next_state[i].append([-1.]*61)
        batch_state=np.asarray(batch_state)
        batch_state=batch_state.astype('float32')
        batch_next_state=np.asarray(batch_next_state)
        batch_next_state=batch_next_state.astype('float32')
        q = self.model.predict(batch_state)
        q_next=self.model.predict(batch_next_state)
        for i in range(self.BATCH):
            q_value = batch_cost[i] + self.gamma * np.max(q_next[i])
            if batch_cost[i] < 0:
                q[i][batch_action_index[i]][batch_action[i]]=batch_cost[i]
            else:
                q[i][batch_action_index[i]][batch_action[i]]=q_value
        batch_state=np.array(batch_state)
        self.model.fit(batch_state,q)
        loss=self.model.evaluate(batch_state,q)
        if self.step_index%10==0:
            self.loss_store.append(loss)
    def train(self):
        index_num=[]
        err_num=[]
        err_n=0
        while True:
            self.rate=self.heap.FinfTop([50,30,20],61)
            s_in=copy.deepcopy(self.state_input)
            for i in range(30):
                s_in[0].insert(0,self.rate[i].tolist())
            res,action_index,action = self.select_action(s_in)
            print("****"+str(action_index)+"****"+str(action))
            old_state, cost= self.step(res,s_in,action_index,action)    
            next_state=copy.deepcopy(self.state_input)
            for i in range(30):
                next_state[0].insert(0,self.rate[i].tolist())
            self.save_store(s_in, action_index,action,cost,next_state)
            if cost==self.max_cost:
                err_n=err_n+1
            if self.step_index!=0:
                err_num.append(err_n*100/self.step_index)
            self.step_index+=1
            print(self.step_index)
            index_num.append(len(self.state_input[0]))
            if self.step_index>2000 and self.step_index<5000:
                self.beta=0.2+0.4*(5000-self.step_index)/3000
#                self.key=0.4+0.3*(self.step_index-2000)/3000
            if self.step_index%20==0 or self.step_index>5000:
                self.experience_replay()
#            if self.step_index==10000:
#                self.model.save('my_model.h5')
            if self.step_index>8000:
 #               plt.plot(np.array(range(len(self.loss_store))),np.array(self.loss_store))
 #               plt.xlabel("experience replay number")
 #               plt.ylabel("loss")
  #              plt.show()
  #              plt.plot(np.array(range(len(index_num))),np.array(index_num))
  #              plt.ylabel("index number")
  #              plt.xlabel("time")
  #              plt.show()
                plt.plot(np.array(range(len(self.reward))),np.array(self.reward))
                plt.ylabel("reward")
                plt.xlabel("time")
                plt.savefig('经验5-1随机1reward.png')
                plt.show()
                for i in range(18):
                    plt.plot(np.array(range(len(self.query_store[i]))),np.array(self.query_store[i]))
                    plt.title("query"+str(i)+" cost")
                    plt.ylabel("cost")
                    plt.xlabel("time")
                    plt.show()
                plt.plot(np.array(range(len(err_num))),np.array(err_num))
                plt.ylabel("error number")
                plt.xlabel("time")
                plt.show()
    def isLegal(self,action_index,action):
        f1 = -1
        f2 = -1
        for i in range(61):
            if self.state_input[0][action_index][i] == 1:
                f1 = i
                break
        for i in range(8):
            if f1>=self.delta[i]:
                f2=i
        if f2 == 7:
            if action>=self.delta[f2]:
                return True
            else:
                return False
        else:
            if action>=self.delta[f2] and action<self.delta[f2+1]:
                return True
            else:
                return False
e=Environment()
e.train()

        