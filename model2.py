import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt


K=input("Enter the total no. of channels: ")
N=input("Enter the total no. of nodes: ")
mean_gain_param=[[0 for j in range(N)] for i in range(K)]
for i in range(K):
    for j in range(N):
        mean_gain_param[i][j]=((i+1.0)*(j+1.0))/((K+1)*(N+1)**2)


T=input("Enter the no. of rounds: ")
npath=input("Enter the total no. of sample paths: ")

freq=[1 for i in range(K)]

def UCB():
    cum_regret_t=[]
    print "UCB Running!"
    alpha=1.5
    for path in range(npath):
        print "Path No.: ",path
        cum_regret=[0.0 for i in range(T)]
        est_gain=[[0.0 for i in range(N)] for j in range(K)] 
        set_chosen_channel=[[[] for j in range(N)] for i in range(K)] 
#        p_t=[1.0/K for i in range(K)]
        decide=[[0.0 for j in range(N)] for i in range(K)]
        total_gain=0.0
        for t in range(K):
#            if t==0:
#                #cum_regret[t]=max(mean_gain_param)-mean_gain_param[t]
#                #print "r=",r
#                #print "###########################",mean_gain_param[0],mean_gain_param[r],cum_regret[t]
#            else:
#                cum_regret[t]=cum_regret[t-1]+(max(mean_gain_param)-mean_gain_param[t])
            #cum_regret[t]=0
            for i in range(N):
                gain=np.random.binomial(1,mean_gain_param[t][i])
                set_chosen_channel[t][i].append(gain)            
            for i in range(K):
                for j in range(N):
                    if(len(set_chosen_channel[i][j])>0):
                        est_gain[i][j]=1.0*np.sum(set_chosen_channel[i][j])/len(set_chosen_channel[i][j])
                    else:
                        est_gain[i][j]=0.0
        for t in range(K,T):
            for i in range(K):
                for j in range(N):
                    decide[i][j]=est_gain[i][j]-np.sqrt(1.0*alpha*np.log(t)/len(set_chosen_channel[i][j]))
            n_k=[]
            for i in range(K):
                k=min(decide[i])
                n_k.append(decide[i].index(k))
            k_max=[]
            for i in range(K):
                f=0
                for l in range(len(set_chosen_channel[i])):
                    f=f+len(set_chosen_channel[i][l])
                k_max.append(est_gain[i][n_k[i]]-np.sqrt(1.0*alpha*np.log(t)/f))
            max_k=max(k_max)
            k_star=k_max.index(max_k)
            freq[k_star]=freq[k_star]+1
#            max_est_gain=max(decide)
#            best_est_arm=decide.index(max_est_gain)
            nk_max=[]
            for i in range(K):
                nk_max.append(est_gain[k_star][n_k[i]])
            max_nk=max(nk_max)
            nk_star=nk_max.index(max_nk)
            gain=np.random.binomial(1,mean_gain_param[k_star][nk_star])
            set_chosen_channel[k_star][nk_star].append(gain)
            total_gain=total_gain+gain
            regret=t*mean_gain_param[k_star][nk_star]-total_gain
            cum_regret[t]=cum_regret[t-1]+regret
            #print "Gain by the chosen arm: ",gain
            #set_chosen_channel[best_est_arm].append(gain)
            #cum_regret[t]=cum_regret[t-1]+(max(mean_gain_param)-mean_gain_param[best_est_arm])
            for i in range(K):
                for j in range(N):
                    est_gain[i][j]=1.0*np.sum(set_chosen_channel[i][j])/len(set_chosen_channel[i][j])
        
        
        cum_regret_t.append(cum_regret)
        
        for i in range(K):
            freq[i]=freq[i]/npath
    
    regret_mean = []
    regret_err = []
    time_epoch=[i for i in range(T)]
    cum_regret_tr=[[0 for i in range(npath)]for j in range(T)]    
    for i in range(T):
        for j in range(npath):
            cum_regret_tr[i][j]=cum_regret_t[j][i]
    freedom_degree = len(cum_regret_tr[0]) - 2
    for regret in cum_regret_tr:
        regret_mean.append(np.mean(regret))
        regret_err.append(ss.t.ppf(0.95, freedom_degree) *ss.sem(regret))
    colors = list("rgbcmyk")
    shape = ['--^', '--d', '--v']
    plt.errorbar(time_epoch, regret_mean, regret_err, color=colors[1])
    plt.plot(time_epoch, regret_mean, colors[1] + shape[1], label='UCB')

UCB()

plt.legend(loc='upper right', numpoints=1)
plt.title("Cumulative Pseudo Regret vs T for T = 25000 and 20 Sample paths")
plt.xlabel("T")
plt.ylabel("Cumulative Pseudo Regret")


plt.clf()
plt.plot(freq)
plt.xlabel("Channels")
plt.ylabel("Average Frequency across 20 sample paths")
plt.title("Plot of Average Frequency against Channels")