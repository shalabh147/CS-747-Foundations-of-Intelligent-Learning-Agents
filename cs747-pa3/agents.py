import numpy as np 
from matplotlib import pyplot as plt
import argparse
import random
from environment import env_action1,env_action2,env_action_stoch

parser = argparse.ArgumentParser(description='agents')
parser.add_argument('--algo', type=str, default=3,help='algo')
parser.add_argument('--actions', type=int, default=4, help='actions')
parser.add_argument('--rows',type=int,help='rows')
parser.add_argument('--columns',type=int,help='columns')
parser.add_argument('--steps',type=int,help='steps')
parser.add_argument('--allthree',type=int,help='three')
parser.add_argument('--stochastic',type=int,help='sto')
args = parser.parse_args()

algo = args.algo
actions = args.actions
n = args.rows
m = args.columns
max_steps = args.steps
all_three = args.allthree
stochastic = args.stochastic

states = n*m



start_st = 30
end_st = 37
state = start_st
episodes = 0
steps = 0
#episode_list = np.zeros(max_steps//20)
#steps_list = []
gamma = 1
epsilon = 0.1
alpha = 0.5
#random_seed_lst = range(0,10,1)
random_seed_lst = [5,10,15,20,25,30,35,40,45,50]



def eps_greedy1(state,Q):
	#print(state)
	greedy_ac = np.argmax(Q[state])
	prob = np.random.rand()
	
	if prob < epsilon:
		
		prob2 = np.random.rand()

		if prob2 < 0.25:
			return 0
		
		if prob2 < 0.5:
			return 1
		
		if prob2 < 0.75:
			return 2
		
		return 3
	

	return greedy_ac

def eps_greedy2(state,Q):
	greedy_ac = np.argmax(Q[state])
	prob = np.random.rand()

	if prob < epsilon:
		prob2 = np.random.rand()
		return min(max(0,(int)(prob2//0.125)),7)


	return greedy_ac


def sarsa():

	steps_list = []
	episode_list = []
	for seed in random_seed_lst:
		np.random.seed(seed)
		state = start_st
		steps = 0
		episodes = 0
		steps_list_temp = []
		episode_list_temp = []
		Q = np.zeros((states,actions))
		if actions == 4:
			action = eps_greedy1(state,Q)
		else:
			action = eps_greedy2(state,Q)
		while True:

			if actions == 4:
				#action = eps_greedy1(state,Q)
				rew,new_state = env_action1(state,action)
				next_ac = eps_greedy1(new_state,Q)
			else:
				
				rew,new_state = env_action2(state,action)

				if stochastic:
					rew,new_state = env_action_stoch(state,action)

					

				next_ac = eps_greedy2(new_state,Q)

			target = rew + gamma*Q[new_state,next_ac]
			Q[state,action] += alpha*(target - Q[state,action])		
			steps += 1
			state = new_state
			action = next_ac
			
			if steps >= max_steps:
				if state == end_st:
					episodes += 1
					state = start_st

				steps_list_temp.append(steps)
				episode_list_temp.append(episodes)
				break

			if steps%1 == 0:
				steps_list_temp.append(steps)
				episode_list_temp.append(episodes)

			if state == end_st:
				state = start_st
				episodes += 1
			#	print(steps)


		steps_list = np.asarray(steps_list_temp)
		episode_list.append(episode_list_temp)
	#print(episode_list)
	episode_list = np.mean(np.asarray(episode_list),axis=0)
	print("Optimal Q", max(Q[start_st]))
	return steps_list,episode_list

	# plt.plot(steps_list,episode_list)
	# plt.show()
	# plt.close()


def expected_sarsa():
	steps_list = []
	episode_list = []
	for seed in random_seed_lst:
		np.random.seed(seed)
		Q = np.zeros((states,actions))
		state = start_st
		steps = 0
		episodes = 0
		steps_list_temp = []
		episode_list_temp = []

		while True:		
			action = eps_greedy1(state,Q)
			rew,new_state = env_action1(state,action)
			prob = np.zeros(actions)

			opt_ac = np.argwhere(Q[new_state] == np.amax(Q[new_state])).flatten()			#distribute 1-epsilon to all the optimal actions
			prob[opt_ac] = (1-epsilon)/len(opt_ac)

			prob = prob + epsilon/4															#distribute epsilon to all 4 actions	

			target = rew + gamma*np.sum(prob*Q[new_state])
			Q[state,action] += alpha*(target - Q[state,action])
			steps += 1
			state = new_state

			if steps >= max_steps:
				if state == end_st:
					episodes += 1
				steps_list_temp.append(steps)
				episode_list_temp.append(episodes)
				break

			if steps%1 == 0:
				steps_list_temp.append(steps)
				episode_list_temp.append(episodes)

			if state == end_st:
				state = start_st
				episodes += 1

		steps_list = np.asarray(steps_list_temp)

		episode_list.append(episode_list_temp)
	#print(episode_list)
	episode_list = np.mean(np.asarray(episode_list),axis=0)

	return steps_list,episode_list




def qlearning():

	steps_list = []
	episode_list = []
	for seed in random_seed_lst:
		np.random.seed(seed)
		Q = np.zeros((states,actions))
		state = start_st
		steps = 0
		episodes = 0
		steps_list_temp = []
		episode_list_temp = []

		while True:
			action = eps_greedy1(state,Q)
			rew,new_state = env_action1(state,action)

			target = rew + gamma*np.max(Q[new_state])
			Q[state,action] += alpha*(target - Q[state,action])
			steps += 1
			state = new_state

			if steps >= max_steps:
				if state == end_st:
					episodes += 1
				steps_list_temp.append(steps)
				episode_list_temp.append(episodes)
				break

			if steps%1 == 0:
				steps_list_temp.append(steps)
				episode_list_temp.append(episodes)

			if state == end_st:
				state = start_st
				episodes += 1


		steps_list = np.asarray(steps_list_temp)

		episode_list.append(episode_list_temp)
	#print(episode_list)
	episode_list = np.mean(np.asarray(episode_list),axis=0)

	return steps_list,episode_list



if all_three:
	steps_list,episode_list = sarsa()
	plt.plot(steps_list,episode_list)
	print("Average Step Size" ,300/(episode_list[-1] - episode_list[-300]))
	#state = start_st
	steps_list,episode_list = expected_sarsa()	
	plt.plot(steps_list,episode_list)
	print("Average Step Size" ,300/(episode_list[-1] - episode_list[-300]))
	#state = start_st
	steps_list,episode_list = qlearning()
	plt.plot(steps_list,episode_list)
	print("Average Step Size" ,300/(episode_list[-1] - episode_list[-300]))
	plt.legend(['sarsa','expected_sarsa','Q-learning'])
	plt.title('Non Stochastic Non King Moves')
	plt.xlabel('Steps')
	plt.ylabel('Episodes')
	plt.show()
	plt.close()
elif algo == "sarsa":
	steps_list,episode_list = sarsa()
	print("Average Step Size" ,300/(episode_list[-1] - episode_list[-300]))
	plt.plot(steps_list,episode_list)
	plt.title('Sarsa(0) Agent Baseline Plot')
	plt.xlabel('Steps')
	plt.ylabel('Episodes')	
	plt.show()
	plt.close()
elif algo == "expected_sarsa":
	steps_list,episode_list = expected_sarsa()
	plt.plot(steps_list,episode_list)
	plt.title('Expected Sarsa Baseline Plot')
	plt.xlabel('Steps')
	plt.ylabel('Episodes')
	plt.show()
	plt.close()
elif algo == "q-learning":
	steps_list,episode_list = qlearning()
	plt.plot(steps_list,episode_list)
	plt.title('Q-Learning Baseline Plot')
	plt.xlabel('Steps')
	plt.ylabel('Episodes')
	plt.show()
	plt.close()

#state = start_st
# stochastic = 0
# actions = 8
# steps_list,episode_list = sarsa()
# plt.plot(steps_list,episode_list)

# stochastic = 1
# actions = 8
# steps_list,episode_list = sarsa()
# plt.plot(steps_list,episode_list)


# stochastic = 0
# actions = 4
# steps_list,episode_list = sarsa()
# plt.plot(steps_list,episode_list)

# plt.title('Sarsa(0) Agent Baseline Plot')
# plt.xlabel('Steps')
# plt.ylabel('Episodes')
# plt.legend(['Non Stochastic, King moves', 'Stochastic, king moves','Non Stochastic, No king moves'])
# plt.show()
# plt.close()