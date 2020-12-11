import numpy as np 
#from matplotlib import pyplot as plt
import argparse
import random
from pulp import *

parser = argparse.ArgumentParser(description='bandit')
parser.add_argument('--mdp', type=str, default=3,help='mdp')
parser.add_argument('--algorithm', type=str, default=3, help='algo')

args = parser.parse_args()

mdp_inst = args.mdp
algo = args.algorithm

file_to_read = mdp_inst


with open(file_to_read,'r') as f:
	for line in f.readlines():
		lst = line.split()
		if lst[0] == "numStates":
			states = int(lst[1])
		elif lst[0] == "numActions":
			actions = int(lst[1])
			T = np.zeros((states,actions,states))
			R = np.zeros((states,actions,states))
		elif lst[0] == "start":
			start_st = int(lst[1])
		elif lst[0] == "end":
			if lst[1] != "-1":
				end_states = lst[1:]
			else:
				end_states = []
		elif lst[0] == "transition":
			s = int(lst[1])
			a = int(lst[2])
			s_prime = int(lst[3])
			T[s,a,s_prime] = float(lst[5])
			R[s,a,s_prime] = float(lst[4])
		elif lst[0] == "mdtype":
			mdptype = lst[1]
		elif lst[0] == "discount":
			gamma = float(lst[1])

#print(T)
#print(R)

def value_iteration():
	V = np.zeros(states)
	old_V = V

	while True:
		X = R + gamma*V
		X = T*X
		#print(X.shape)
		V = np.max(np.sum(X,axis=-1),axis=-1)

		if abs(np.linalg.norm(V) - np.linalg.norm(old_V)) < 1e-12:
			break

		old_V = V

	X = R + gamma*V
	X = T*X
	Q = np.sum(X,axis=-1)

	pi = np.argmax(Q,axis=-1)
	return V,pi

#print(value_iteration())
def solver(policy):

	A = np.zeros((states,states))
	B = np.zeros(states)
	# for i in range(states):
	# 	for j in range(states):
	# 		if i == j:
	# 			A[i,j] = 1 - gamma*T[i,policy[i],i]
	# 		else:
	# 			A[i,j] = -1*gamma*T[i,policy[i],j]

	A = np.eye(states) - gamma*T[np.arange(states),policy,:]
	TR = np.multiply(T,R)
	TR_sum = np.sum(TR,axis=-1)
	#for s in range(states):
	#	B[s] = TR_sum[s,policy[s]]
	B = TR_sum[np.arange(states),policy]
	# A = np.eye(states)
	# print(T[:,policy,:].shape)
	# A = A - T[:,policy,:]
	# TR = np.multiply(T,R)
	# TR_sum = np.sum(TR,axis=-1)
	# B = TR_sum[:,policy]
	# print(B.shape)
	V = np.linalg.solve(A,B)

	X = R + gamma*V
	X = np.multiply(T,X)
	Q = np.sum(X,axis=-1)

	return V,Q



def policy_iteration():
	pi = np.random.randint(0,actions,(states))
	#print(pi)
	pi_dash = pi
	while True:
		V_pi,Q_dash = solver(pi)
		#print(V_pi)
		
		# for s in range(states):
		# 	for ac in range(actions):
		# 		if pi[s]!=ac:
		# 			if Q_dash[s,ac] > V_pi[s]:
		# 				pi_dash[s] = ac

		pi_dash = np.argmax(Q_dash,axis=-1)

		if np.all(pi == pi_dash):
			break

		pi = pi_dash

	return V_pi,pi

def linear_prog():
	prob = LpProblem("mdp",LpMaximize)
	V_var = []

	for s in range(states):
		name = "s" + str(s)
		x1=LpVariable(name)
		V_var.append(x1)

	#for s in range(states):
	#	prob += -1*V_var[s]
	num_Vvar = np.asarray(V_var)
	prob += -1*np.sum(num_Vvar)
	#prob += pulp.lpSum([-1*V_var[s] for s in range(states)])
	for s in range(states):
		for ac in range(actions):
			prob += np.sum(T[s,ac]*R[s,ac] + gamma*T[s,ac]*num_Vvar) <= num_Vvar[s]
			#prob += pulp.lpSum([T[s,ac,x]*R[s,ac,x] + gamma*T[s,ac,x]*V_var[x] for x in range(states)]) <= V_var[s]
	

	result = prob.solve(PULP_CBC_CMD(msg=0))
	
	#print(prob.variables())
	#print(V_var)
	V_opt = []
	for v in V_var:
		V_opt.append(v.varValue)

	V_opt = np.asarray(V_opt)
	X = R + gamma*V_opt
	X = np.multiply(T,X)
	Q = np.sum(X,axis=-1)

	pi = np.argmax(Q,axis=-1)
	return V_opt,pi

		#print(v.name, "=", v.varValue)
		

if algo == "vi":
	Opt_V,Opt_pi = value_iteration()
elif algo == "hpi":
	Opt_V,Opt_pi = policy_iteration()
elif algo == "lp":
	Opt_V,Opt_pi = linear_prog()


for s in range(states):
	print(str(Opt_V[s]) + "\t" + str(Opt_pi[s]))