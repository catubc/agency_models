import matplotlib.pyplot as plt
import numpy as np
import scipy

from tqdm import trange

#
def parallel_simulations(token):

    h = Human()

    #
    a = Agent()
    a.human = h

    # generate world object
    w = World(a, h)

    # Update human params
    h.random_init = False
    h.agent_scaling = 1/1000
    h.noise_self_scaling = 0 #1/1000
    h.noise_world_scaling = 1/10
    h.action_scaling = 0 #1/1000
    h.option_depletion_max = 0  # max amount of depletion alowed per action
    h.agency_recovery_rate = 1.1  # rate at which agency recovers
    h.verbose = False
    h.text = prefix
    h.agent = a
    h.w = w

    #
    h.n_epochs = 10000

    #
    h.run_simulation(token)

#
class World():
    # ts1 
    def __init__(self, agent, human):
        
        self.agent = agent
        self.human = human

    # test
    def update_world(self):
        
        # add some noise to the agent's values
        self.noise_world = np.random.rand(len(self.human.action_types))-0.5
        #self.noise_world = np.random.rand(len(self.human.action_types))
        
        #
        return self.noise_world

#
class Human():
    
    #
    def __init__(self, 
                 ):

        # 
        self.params = [
                        [0.1,1],
                        [2,10],
                        [5,10],
                        [10,80],
                    ]

        #
        self.verbose=False

        #
        self.setup_actions()

        #
        self.clrs = ['darkorchid','blue','red','green']

        #
        self.linewidth=5

        # self.agent = agent
        # self.random_init = random_init
        
        # #
        # self.agent_scaling = agent_scaling
        # self.noise_self_scaling = noise_self_scaling
        # self.action_scaling = action_scaling
        # self.noise_world_scaling = noise_world_scaling
        
        #
        self.make_dists()
        
    
    def visualize_dists(self):

        print ("MUST THINK ABOUT ADDING LESS PREDICTABLE HUMANS <- and how human agency/predictablity is the key problem..")

        #
        plt.figure(figsize=(10,5))
        width = 0.01
        x = np.arange(0,1+width,width)

        #
        ctr=0
        for param in self.params:
            
            #
            dist = scipy.stats.beta(param[0],param[1])
            
            #
            distp = dist.pdf(x)
            distp[distp == np.inf]=np.nan
            
            dist = scipy.stats.beta(param[0],param[1])

            #
            value = 1./dist.expect()


            ############################################
            ############################################
            ############################################
            ax1 = plt.subplot(121)
            
            #    
            plt.plot(x,distp, label="option: "+str(ctr)+
                    ", value = "+str(round(value,2)),
                    linewidth=self.linewidth,
                    c=self.clrs[ctr])
            
            #
            E = param[0]/(param[0]+param[1])
            expect = dist.expect()
            plt.plot([expect,expect],[0,13],'--',
                    c=self.clrs[ctr])
            plt.ylim(bottom=0)
            #plt.semilogy()
            plt.legend(title='probability of reward for each option '
                    + "\n dash = expected; value = 1/expected",
                    fontsize=10)

            ############################################
            ############################################
            ############################################
            ax1 = plt.subplot(122)
            
            #y = distp.cumsum()
            if distp[0]==np.nan:
                distp[0]=0
            
            #
            y = np.nancumsum(distp)
            plt.plot(x,y/y[-1], label="option: "+str(ctr),
                    c=self.clrs[ctr])
            #


            ctr+=1
            
        # plt.ylabel("Prob(reward)")
        # plt.xlabel("Reward")
        plt.savefig('/home/cat/agency_fig1.svg')
        plt.ylim(bottom=0)
        plt.show()

    #       
    def run_simulation(self, save_token=None):

        #
        actions = []
        values = []
        tot_vals = []
        tot_rewards = []
        if self.verbose:
            for k in trange(self.n_epochs, desc="Running simulation"):
                self.select_action()
                actions.append(self.action_probs)
                values.append(self.vals)
                tot_vals.append(self.vals.sum())
                tot_rewards.append(self.reward)
        else:
            for k in range(self.n_epochs):
                self.select_action()
                actions.append(self.action_probs)
                values.append(self.vals)
                tot_vals.append(self.vals.sum())
                tot_rewards.append(self.reward)

        #
        self.actions = np.vstack(actions)
        self.values = np.vstack(values)
        self.total_rewards = np.vstack(tot_rewards)

        # 
        if save_token is not None:
            np.savez('/home/cat/agency_21st_century_paper/data/'+
                    self.prefix+ '_agent_'+str(save_token)+'.npz', 
                    actions = self.actions,
                    values = self.values,
                    total_rewards = self.total_rewards)

    #
    def plot_simulation(self):
        #######################################
        #
        plt.figure(figsize=(10,10))
        ax=plt.subplot(221)
        for k in range(4):
            plt.plot(self.actions[:,k], 
                     c=self.clrs[k],
                     linewidth=self.linewidth,
                     label="option "+str(k))

        #
        plt.plot([0,self.n_epochs], 
                [.25,.25],
                '--',
                c='grey',
                alpha=1,
                linewidth=self.linewidth)

        #
        plt.ylim(0,1)
        plt.legend(fontsize=12)
        plt.xlabel("Time")
        plt.ylabel("Probability of choosing action for human")

        #############################################
        ax=plt.subplot(222)
        for k in range(4):
            plt.plot(self.values[:,k]*self.expectations[k], 
                    c=self.clrs[k],
                    linewidth=self.linewidth,
                    label=self.action_types[k])
            
        plt.plot([0,self.n_epochs], [1.0,1.0],'--',c='grey',
                  alpha=1,
                  linewidth=self.linewidth)
        #plt.plot(tot_vals, c='orange')
        plt.ylabel("Human value for action")
        plt.xlabel("Time")
        plt.ylim(bottom=0)

        #############################################
        ax=plt.subplot(223)
        self.total_rewards = np.sum(self.total_rewards, axis=0)
        for k in range(4):
            plt.bar(k,self.total_rewards[k],0.9,
                    color=self.clrs[k])
            
        #plt.plot(tot_vals, c='orange')
        plt.ylabel("reward received for simulation")
        plt.xlabel("Time")
        plt.ylim(bottom=0)

        #############################

        #
        plt.savefig('/home/cat/agency_fig3.svg')

        #
        plt.show()
        
    #
    def make_dists(self):

        #
        self.dists = []
        self.expectations = []
        self.vals = []
        self.starting_vals = []
        
        #
        ctr=0
        self.all_value_sum = 0
        for param in self.params:

            #
            dist = scipy.stats.beta(param[0],param[1])
            
            #
            self.dists.append(dist)
            
            #
            self.expectations.append(dist.expect())

            #
            value = 1./dist.expect()

            #
            self.all_value_sum+=value
            
            #
            if ctr == 100:
                value = value*2.2
            
            #
            self.vals.append(value)
            
            # 
            self.starting_vals.append(value)
            
            #
            ctr+=1

        #
        self.expectations = np.array(self.expectations)
        self.vals = np.array(self.vals)
        if self.verbose:
            print ("starting human vals; ", self.starting_vals)

    # 
    def setup_actions(self):
        
        # options (option depletion)
        self.action_types = ["salad", "oatmeal", "soup", "chocolate"]
        self.action_types = ['explore','play','nothing','solve_problem']
        
    # better named as forward 
    def select_action(self):
        
        # get agent rec
        # print ("interacting with agent")
        self.interact_agent()
       
        #
        #print ("interacting with world")
        self.interact_world()
    
        # update action probs
        #print ("updating human values")
        self.update_human_values_pre_action()
        
        # select an action by mulitplying value by expectation
        self.action_probs = self.vals*self.expectations

        #
        if False:
            for k in range(len(self.action_probs)):
                if self.action_probs[k]<0:
                    self.action_probs[k] = 0

            # if np.min(self.action_probs)<0:
            #     #print ("neative prob: ", self.action_probs)
            #     self.action_probs = self.action_probs-np.min(self.action_probs)
        
        # then normalizing 
        self.action_probs = self.action_probs/np.sum(self.action_probs)

        # for now we just assume all starting vals are the same
        ctr=0
        while np.min(self.action_probs)<(0.25*self.option_depletion_max):
            # if print_flag:
            #     print ("depleting options: ", self.action_probs)
            #     print_flag = False

            #
            idx = np.argmin(self.action_probs)

            # change probabilities directly
            if False:
                self.action_probs[idx] = self.action_probs[idx]*self.agency_recovery_rate

            # change the vals directly
            else:
                self.vals[idx] = self.vals[idx]*self.agency_recovery_rate
                self.action_probs = self.vals*self.expectations

            # renormalize    
            self.action_probs = self.action_probs/np.sum(self.action_probs)
            #print ("depleting options: ", self.action_probs)

        #print ("# steps: ", ctr)

        self.action = np.random.choice(np.arange(len(self.action_types)), 
                                                 p = self.action_probs)
        
        #
        if self.verbose:
            print ("updated action probs: ", self.action_probs)
            print ("updated human vals: ", self.vals)
            #print ('self.expectations: ', self.expectations)
 
        # sample the action to see if successful
        prob_success = self.dists[self.action].rvs(1)
        rnd = np.random.rand(1)
        self.reward = np.zeros(len(self.vals))
        if rnd<prob_success:
            self.reward[self.action] = self.vals[self.action] 

        # make an action selection
        # print ("agent rec: ", self.agent_rec, " human action: ", self.action, ", probs: ", self.action_probs)
    
#     #
#     def update_human_values_post_action(self):
        
#         #
#         self.vals[self.action] += self.action_scaling
        
#         # renormalize to keep the same amount of value in the world
#         self.vals = self.vals/self.vals.sum()*self.all_value_sum

            
    #
    def update_human_values_pre_action(self):
        
        # agent generated influence
        self.vals = self.vals + self.agent_recommendation*self.agent_scaling
        if self.verbose:
            print ("self vals: ", self.vals)

        # self generated noise/influence; set to zero for now
        self.noise_human = np.random.rand(len(self.action_types))
        self.vals = self.vals+self.noise_human*self.noise_self_scaling
        if self.verbose:
            print ("self vals: ", self.vals)
        
        # world random walk influence
        self.vals = self.vals+self.noise_world*self.noise_world_scaling
        if self.verbose:
            print ("self vals: ", self.vals)

        # here we ensure that the values are always positive
        if True:
            for k in range(len(self.vals)):
                if self.vals[k]<0:
                    self.vals[k] = 0

        #############
        # renomralize so that we only get max original total value:
        self.vals = self.vals/self.vals.sum()*self.all_value_sum

       
        # 
        if self.verbose:
            print ("self vals last: ", self.vals)
            print ("")
    #
    def interact_world(self):
        
        #
        self.noise_world = self.w.update_world()
               
    #
    def interact_agent(self):
        
        #
        self.agent_recommendation = self.agent.update_action_values(self)
        
        #
        if self.verbose:
            print ("agent valuation ; ", self.agent_valuation)

#
class Agent():
    
    #
    def __init__(self):
        self.verbose = False
        pass
    
    #
    def setup_actions(self):
        
        pass
    
    #
    def update_action_values(self, human):
        
        from scipy.special import logit, expit, softmax
        
        # we sample from each distrubiont and mulitiply by expected val.
        #print ("human vals; ", human.vals)
        agent_valuation = []
        for k in range(len(human.dists)):
            
            # 
            sample = human.dists[k].rvs(1)
            
            # 
            if human.use_starting_vals==False:
                agent_valuation.append(human.vals[k]*sample)
            else:
                agent_valuation.append(human.starting_vals[k]*sample)
        
        #
        agent_valuation = np.array(agent_valuation)
        
        # 
        if False:
            agent_valuation = agent_valuation/np.sum(agent_valuation)
        else:
            temp = np.zeros(4)
            temp[np.argmax(agent_valuation)] = 1
            agent_recommendation = temp
            
        #
        agent_recommendation = agent_recommendation.squeeze()

        #
        return agent_recommendation
    
    #
    def update_world_state(self):
        
        pass