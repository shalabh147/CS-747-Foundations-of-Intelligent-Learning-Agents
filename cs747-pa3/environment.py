import numpy as np 
#from matplotlib import pyplot as plt
import argparse
import random

wind = np.array([0,0,0,1,1,1,2,2,1,0])
n = 10
m = 7
start_st = 30
end_st = 37

def env_action1(state,action):
	y = state//n
	x = state%n
	shift = wind[x]


	if action == 0:														#move up
		y = min(max(0,y-1-shift),m-1)
	elif action == 1:													#move down
		y = max(0,min(y+1-shift,m-1))
	elif action == 2:													#move left
		x = max(0,x-1)
		y = max(y-shift,0)
	elif action == 3:													#move right
		x = min(x+1,n-1)
		y = max(y-shift,0)

	#print(x,y)
	new_state = (int)(y*n + x)

	return -1,new_state


def env_action2(state,action):
	y = state//n
	x = state%n
	shift = wind[x]

	if action == 0:														#move up
		y = min(max(0,y-1-shift),m-1)
	elif action == 1:													#move down
		y = max(0,min(y+1-shift,m-1))
	elif action == 2:													#move left
		x = max(0,x-1)
		y = max(y-shift,0)
	elif action == 3:													#move right
		x = min(x+1,n-1)
		y = max(y-shift,0)
	elif action == 4:													#move to left up
		y = min(max(0,y-1-shift),m-1)
		x = max(0,x-1)
	elif action == 5:													#move right up
		y = min(max(0,y-1-shift),m-1)
		x = min(x+1,n-1)
	elif action == 6:													#move left down
		y = max(0,min(y+1-shift,m-1))
		x = max(0,x-1)
	elif action == 7:													#move right down
		y = max(0,min(y+1-shift,m-1))
		x = min(x+1,n-1)

	new_state = (int)(y*n+x)

	return -1,new_state


def env_action_stoch(state,action):
	y = state//n
	x = state%n
	shift = wind[x]
	stoch = False

	if shift:				#stochastic only if there is non zero wind in the column
		stoch = True		


	if action == 0:														#move up
		y = min(max(0,y-1-shift),m-1)
	elif action == 1:													#move down
		y = max(0,min(y+1-shift,m-1))
	elif action == 2:													#move left
		x = max(0,x-1)
		y = max(y-shift,0)
	elif action == 3:													#move right
		x = min(x+1,n-1)
		y = max(y-shift,0)
	elif action == 4:													#move to left up
		y = min(max(0,y-1-shift),m-1)
		x = max(0,x-1)
	elif action == 5:													#move right up
		y = min(max(0,y-1-shift),m-1)
		x = min(x+1,n-1)
	elif action == 6:													#move left down
		y = max(0,min(y+1-shift,m-1))
		x = max(0,x-1)
	elif action == 7:													#move right down
		y = max(0,min(y+1-shift,m-1))
		x = min(x+1,n-1)

	if stoch == False:
		new_state = (int)(y*n+x)
		return -1,new_state

	prob = np.random.rand()

	if prob < 1/3:		
		new_state = (int)(y*n+x)
		return -1,new_state

	if prob < 2/3:
		y = max(y-1,0)
		new_state = (int)(y*n+x)
		return -1,new_state

	if prob < 1:
		y = min(y+1,m-1)		
		new_state = (int)(y*n+x)
		return -1,new_state

	






