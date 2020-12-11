import numpy as np 
from matplotlib import pyplot as plt
import argparse
import random

parser = argparse.ArgumentParser(description='bandit')
parser.add_argument('--instance', type=str, default=3,help='bandit instance')
parser.add_argument('--algorithm', type=str, default=3, help='algo')
parser.add_argument('--randomSeed', type=int, default=3, help='seed')
parser.add_argument('--epsilon', type=float, default=3, help='eps')
parser.add_argument('--horizon', type=int, default=3, help='T')


args = parser.parse_args()

randomseed = args.randomSeed
inst = args.instance
algorithm = args.algorithm
epsilon = args.epsilon
horizon = args.horizon

np.random.seed(randomseed)
inst_lst = []
instance_file = inst
with open(instance_file,'r') as f:
	for line in f.readlines():
		lst = line.split("\n")
		inst_lst.append(float(lst[0]))

num_arms = len(inst_lst)
inst_lst = np.asarray(inst_lst)
num_pulls = np.zeros(num_arms)
tot_reward = np.zeros(num_arms)
mean_reward = np.zeros(num_arms)

def multi_arm_bandit(arm=False):
	reward = 0

	prob_reward = inst_lst[arm]
	#random.seed(randomseed)
	prob = np.random.rand()

	if prob > prob_reward:
		reward = 0
	else:
		reward = 1

	return reward

	
	

def epsilon_greedy(epsilon):
	#num_arms = len(inst)
	for i in range(num_arms):
		arm = i
		rew = multi_arm_bandit(arm)
		num_pulls[arm] += 1
		tot_reward[arm] += rew
		mean_reward[arm] = tot_reward[arm]/num_pulls[arm]

	for t in range(num_arms,horizon):
		arm = -1
		#random.seed(randomseed)
		prob = np.random.rand()
		if prob > epsilon:
			arm = np.argmax(mean_reward)
		else:
			arm = np.random.randint(0,num_arms)


		reward = multi_arm_bandit(arm)
		num_pulls[arm] += 1
		tot_reward[arm] += reward
		mean_reward[arm] = tot_reward[arm]/num_pulls[arm]
		

def UCB():
	ucb_arr = np.zeros(num_arms)
	
	for _ in range(1):
		for i in range(num_arms):
			arm = i
			rew = multi_arm_bandit(arm)
			num_pulls[arm] += 1
			tot_reward[arm] += rew
			mean_reward[arm] = tot_reward[arm]/num_pulls[arm]

	for t in range(1*num_arms,horizon):
		for i in range(num_arms):
			ucb_arr[i] = mean_reward[i] + np.sqrt((2*np.log(t))/num_pulls[i])

		arm = np.argmax(ucb_arr);
		rew = multi_arm_bandit(arm);
		num_pulls[arm] += 1
		tot_reward[arm] += rew
		mean_reward[arm] = tot_reward[arm]/num_pulls[arm]


def KL(p,q):
	if p == 0:
		p = 1e-5

	if p == 1:
		p = 1 - 1e-5

	if q == 0:
		q = 1e-5

	if q == 1:
		q = 1 - 1e-5

	return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))


def bs(pull,emp,t,c):
	exp = np.log(t) + c*np.log(np.log(t))
	ans = emp
	lo = emp
	hi = 1.0
	mid = (lo + hi)/2
	#print("mid",mid)
	#print("emp",emp)
	#print("KL",KL(emp,mid))
	
	while lo < hi - 1e-3:
		mid = (lo + hi)/2

		if pull*(KL(emp,mid)) <= exp:
			lo = mid
		else:
			hi = mid
		
	if(pull*KL(emp,mid) <= exp):
		return mid
	
	return lo


def KL_UCB():
	klucb_arr = np.zeros(num_arms)
	for _ in range(1):
		for i in range(num_arms):
			arm = i
			rew = multi_arm_bandit(arm)
			num_pulls[arm] += 1
			tot_reward[arm] += rew
			mean_reward[arm] = tot_reward[arm]/num_pulls[arm]

	
	for t in range(1*num_arms,horizon):
		for i in range(num_arms):
			q = bs(num_pulls[i],mean_reward[i],t,3)
			#print(q)
			klucb_arr[i] = q
		arm = np.argmax(klucb_arr);
		rew = multi_arm_bandit(arm);
		num_pulls[arm] += 1
		tot_reward[arm] += rew
		mean_reward[arm] = tot_reward[arm]/num_pulls[arm]		


def thompson_sampling():
	succ_arr = np.zeros(num_arms)
	fail_arr = np.zeros(num_arms)
	thomp_arr = np.zeros(num_arms)
	for t in range(horizon):
		for i in range(num_arms):
			a = succ_arr[i]+1
			b = fail_arr[i]+1
			thomp_arr[i] = np.random.beta(a,b)

		arm = np.argmax(thomp_arr)
		rew = multi_arm_bandit(arm)
		if rew == 0:
			fail_arr[arm] += 1
		else:
			succ_arr[arm] += 1

		num_pulls[arm] += 1
		tot_reward[arm] += rew
		mean_reward[arm] = tot_reward[arm]/num_pulls[arm]

def thompson_sampling_hint(hints):
	likelihood_mat = np.ones((num_arms,num_arms))
	#likelihood_mat = likelihood_mat*(1e12)
	succ_arr = np.zeros(num_arms)
	fail_arr = np.zeros(num_arms)

	for _ in range(1):
		for i in range(num_arms):
			arm = i
			rew = multi_arm_bandit(arm)
			if rew == 0:
				fail_arr[arm] += 1
				likelihood_mat[arm] = np.multiply(likelihood_mat[arm],1-hints)
			else:
				succ_arr[arm] += 1
				likelihood_mat[arm] = np.multiply(likelihood_mat[arm],hints)

			num_pulls[arm] += 1
			tot_reward[arm] += rew
			mean_reward[arm] = tot_reward[arm]/num_pulls[arm]
			likelihood_mat = likelihood_mat/np.sum(likelihood_mat,axis=1,keepdims=True)
	#sum_of_rows = np.sum(likelihood_mat,axis=1,keepdims=True)
	#inverted = 1/sum_of_rows
	#ikeli_mat = np.multiply(likelihood_mat,inverted)
	

	for t in range(num_arms*1,horizon):
		X = np.argmax(likelihood_mat,axis=0)
		arm = X[-1]

		rew = multi_arm_bandit(arm)
		#print(arm,rew)
		if rew == 0:
			fail_arr[arm] += 1
			likelihood_mat[arm] = np.multiply(likelihood_mat[arm],1-hints)
		else:
			succ_arr[arm] += 1
			likelihood_mat[arm] = np.multiply(likelihood_mat[arm],hints)

		#likelihood_mat = 1e12*likelihood_mat

		num_pulls[arm] += 1
		tot_reward[arm] += rew
		mean_reward[arm] = tot_reward[arm]/num_pulls[arm]

		#sum_of_rows = np.sum(likelihood_mat,axis=1,keepdims=True)
		#inverted = 1/sum_of_rows
		#likeli_mat = likelihood_mat/sum_of_rows
		likelihood_mat = likelihood_mat/np.sum(likelihood_mat,axis=1,keepdims=True)
	#print(likelihood_mat)




if algorithm == "epsilon-greedy":
	epsilon_greedy(epsilon)
elif algorithm == "ucb":
	UCB()
elif algorithm == "kl-ucb":
	KL_UCB()
elif algorithm == "thompson-sampling":
	thompson_sampling()
elif algorithm == "thompson-sampling-with-hint":
	hint_ls = np.sort(inst_lst)
	#print(inst_lst)
	#print(hint_ls)
	thompson_sampling_hint(hint_ls)


cumulative_reward = np.sum(tot_reward)
max_mean = np.max(inst_lst)
cumulative_regret = horizon*max_mean - cumulative_reward

print("{}, {}, {}, {}, {}, {}".format(instance_file,algorithm,randomseed,epsilon,horizon,cumulative_regret))

#def UCB()


#print(multi_arm_bandit(inst))
