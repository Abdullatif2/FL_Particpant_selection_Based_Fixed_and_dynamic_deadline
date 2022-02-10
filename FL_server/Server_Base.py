
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from operator import itemgetter
from itertools import groupby
import csv
import gurobipy as gp
from gurobipy import GRB
from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad
from mip import Model, xsum, maximize, BINARY,CONTINUOUS
import h5py
import random

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);
        self.parameters = params
        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)
        self.clients = self.setup_clients(dataset, self.client_model)
        self.dataset=dataset
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.get_params()

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc,self.lowerbound, self.upperbound, self.proposed_utlization = [], [], [],[],[],[]

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses


    def show_grads(self):  
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)  

        intermediate_grads = []
        samples=[]

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model) 
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(client_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples)) 
        intermediate_grads.append(global_grads)

        return intermediate_grads
 
  
    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self, prox=False, lamb=0, learning_rate=0, data_set="", num_users=0, batch=0):
        alg = data_set 
        alg = alg + "_" + str(learning_rate) + "_" + str(num_users) + "u" + "_" + str(self.batch_size) + "b"
        with h5py.File("./results/"+'{}_{}.h5'.format(alg, self.parameters['num_epochs']), 'w') as hf:
            hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
            hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            hf.close()
        # pass

    def select_clients(self, round, model_size, Flops, num_clients=20):
        np.random.seed(round) 
        Ptx_Max = 0.1
        Ptx_Min = 0.0001 #0.2
        N0  = 1e-10    #    -> Decrease BW 
        BW  = 1e6     #Mhz -> Increase BW -
        msize=model_size
        print('Model size is :',msize)
        #number_of_subchannels=(BW//msize)
        T_frame=3000
        f_max=1.5 #GHz

        f_min=0.3 #GHz


        #print('model size:',msize,'Subchannels:',number_of_subchannels)
        data_size=[None]*len(self.clients)
        comp=[None]*len(self.clients)
        users, groups, train_data, test_data = self.dataset
        REUSED_TRAFFIC=False
        num_of_clients= len(users)
        gain_list, ratios= mobile_gen(num_of_clients,REUSED_TRAFFIC)
        T_com_max = msize/(BW*np.log2(Ptx_Min/np.mean(ratios) + 1))
        T_com_min = msize/(BW *np.log2(Ptx_Max/np.mean(ratios) +1))
        Deadline_com = (T_com_min + T_com_max)/2
        Pwr=[np.random.uniform(Ptx_Min, Ptx_Max) for k in range(num_of_clients)]
        tau = [msize/(BW *np.log2(Pwr[k]/ratios[k] +1)) for k in range(num_of_clients)]

        for i in range(len(users)):
            data_size[i]=len(train_data[users[i]]['x'])
            comp[i] = (self.num_epochs * (data_size[i])//self.batch_size) * self.batch_size * Flops
        T_total= [comp[k]+tau[k] for k in range(num_of_clients)]

        Deadline_comp = max(comp)
        Deadline = Deadline_comp+Deadline_com
        client=[]
        client2=[]
        #print('Data Size: ', data_size)
        for k in range(num_of_clients):
            client.append({'Id':k,'Num_Samples':data_size[k],'T_cmp':comp[k],'T_comm':tau[k], 'total_time':T_total[k]})
            client2.append({'Id':k,'Num_Samples':data_size[k],'T_cmp':comp[k],'T_comm':tau[k], 'total_time':T_total[k]})   
        client.sort(key=itemgetter('Num_Samples'),reverse=True)
        selected=[]
        selected1=[]
        S=np.zeros(len(users))
        num_clients = min(num_clients, len(self.clients))
        for i,c in enumerate(client):
            if c['total_time'] < Deadline and sumtaue1(selected,tau, tau[c['Id']])<=T_frame:
                S[i]=1
                selected.append(c)
                selected1.append(c['Id'])   

        # ///////////////////////////   Optmizarion ////////////////////////
        print("Heuristic Lower Bound is:",np.dot(S,data_size))
        #m = Model('ParticipantSelection')
        n1= range(len(self.clients))
        tau1=dict(zip(n1, tau))
        weights1=dict(zip(n1, data_size))
        Tcomp1=dict(zip(n1, comp))
        Total1=dict(zip(n1, T_total))
        Total2=dict(zip(n1, T_total/Deadline))
        m = gp.Model("Client selection") 



       #x = [m.add_var('x',lb = 0, ub =1,var_type=CONTINUOUS) for i in I]
        #x = [m.add_var('x',var_type=BINARY) for i in I]
        x = m.addVars(num_of_clients, vtype=GRB.CONTINUOUS)
        # f = m.addVars(num_of_clients, vtype=GRB.CONTINUOUS)
        # p = m.addVars(num_of_clients, vtype=GRB.CONTINUOUS)
        m.ModelSense = GRB.MINIMIZE
        m.addConstr(sum(x[i]*tau1[i] for i in n1) <= T_frame)
        for i in n1:
            m.addConstr(x[i]*Total1[i] <= Deadline)
        m.addConstrs(x[i]>=0 for i in n1)
        m.addConstrs(x[i] <=1 for i in n1)
        # m.addConstrs(Ptx_Min <= p  <= Ptx_Max for i in n1)
        # m.addConstrs(f_min <= f  <= f_max for i in n1)
        m.addConstrs(x[i] <=1 for i in n1)
        m.setObjective(x.prod(Total2) - x.prod(weights1))
        m.update()
        m.optimize()
        selected = [i for i in n1 if x[i].x >= 0]
        clientrelax=[]

        for k in n1:
            clientrelax.append({'Id':k,'data_size':data_size[k],'Priority':x[k].x,'uploadT':tau[k]})
            clientrelax.sort(key=itemgetter('Priority'),reverse=True)
        data_S=[]
        selected_Participant=[]
        selected_Participant1=[]
        S2=np.zeros(len(clientrelax))
        for i,c in enumerate(clientrelax):
            if Totaluploadingtime(selected_Participant,tau, tau[c['Id']])<=T_frame:
                data_S.append(c['data_size'])
                S2[i]=1
                selected_Participant.append(c)
                selected_Participant1.append(c['Id'])

        self.lowerbound.append(sum(S*data_size))
        self.upperbound.append(m.ObjBound) 
        self.proposed_utlization.append(sum(S2*data_size))
        print("The optimal number of participants to select is:",len(selected_Participant),"Out of ",len(client))
        print('Total Uploading time is:',sum(tau))
        su=0
        for i,c in enumerate(selected_Participant):
            su+=selected_Participant[i]['uploadT']  
        print('Total is: ',su)        
        return selected_Participant1, np.asarray(self.clients)[selected_Participant1],data_size


    def aggregate(self, wsolns, weighted=True):
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])
        for (w, soln) in wsolns:  # w is the number of samples
            # Equal weights
#            if(weighted==False):
#                w=1 # Equal weights
            # print('Weights is:', w)
            # w= random.randint(100,1000)
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w*v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]
        return averaged_soln

    def aggregate_derivate(self, fsolns, weighted=True):
        total_derivative = 0.0
        base = [0]*len(fsolns[0][1])
        for (f, soln) in fsolns:  # w is the number of samples
            total_derivative += f
            for i, v in enumerate(soln):
                base[i] += f*v.astype(np.float64)

        averaged_soln = [v / total_derivative for v in base]
        return averaged_soln
def to_watt(x):
    return 10**(x / 10)

def to_dBW(x):
    return 10*math.log10(x)

### Exponential distribution of channel_gain
def exp_gain(dist, mode = 0):
    g0 = -40 #dB
    g0_W = to_watt(g0)
    d0 = 1 #1 m
    mean = g0_W*((d0/dist)**4)

    if (mode == 0):
        gain_h = np.random.exponential(mean)
    else:
        gain_h = mean
#     print("here1: ", dist, " : ", mean, " : ",gain_h)
    return gain_h #in Watts

def noise_per_gain(gain,N0):  #N0/h
    return N0/gain

# def shanon_capacity():
#     return
def fedcs():
    return 5
def mobile_gen(num_of_clients,REUSED_TRAFFIC):
    if(REUSED_TRAFFIC):
        print("Reused data")
    else:
        Ptx_Max = 1.
        Ptx_Min = 0.0001 #0.2
        N0  = 1e-10    #    -> Decrease BW 
        BW  = 1e6     #Mhz -> Increase BW -
        Distance_min = 2  #2m
        Distance_max = 500 #50m
        Distance_avg   = (Distance_max + Distance_min)/2.
        dist_list = np.zeros(num_of_clients)
        gain_list = np.zeros(num_of_clients)
        ratios    = np.zeros(num_of_clients)
        dist_list[:] = np.random.uniform(Distance_min,Distance_max,num_of_clients)
        for n in range(num_of_clients):
            gain_list[n] = exp_gain(dist_list[n])
            ratios[n]    = noise_per_gain(gain_list[n],N0)
        return  gain_list, ratios
def sumtaue1(S,tau,tau1):
    total=0
    for i,item in enumerate(S):
        k = int(item['Id'])
        #print(k)
        total+=tau[k]
    return  total+tau1
def Totaluploadingtime(S,tau,tau1):
    total=0
    for i,item in enumerate(S):
        k = int(item['Id'])
        #print(k)
        total+=tau[k]
    return  total+tau1    

