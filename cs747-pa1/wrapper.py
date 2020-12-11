import matplotlib.pyplot as plt
import numpy as np
import os

seed = range(0,50)
horizon = [100,400,1600,6400,25600,102400]
#horizons = [100,400]
algorithms = ['epsilon-greedy','ucb','kl-ucb','thompson-sampling']
instance2 = ['../instances/i-2.txt']
instance1 = ['../instances/i-1.txt']
instance3 = ['../instances/i-3.txt']
eps = 0.02



horizon=102400
#dom=np.linspace(0,0.02,20)
#dom = list(dom)
dom = [0.001,0.01,0.05]
ans=[]
reg_list=[]
#print(in_file)
#f1 = open(in_file, 'r')
# means = [float(line.strip()) for line in f1.readlines()]
os.system("rm out.txt")
for i in (range(len(dom))):
	eps=dom[i]
	reg=0
	Reg = 0
	for j in range(50):
		s = "python3 bandit.py --instance ../instances/i-3.txt --algorithm epsilon-greedy" + " --randomSeed " + str(j) + " --epsilon " + str(eps) + " --horizon 102400 > out.txt"
		os.system(s)
		with open('out.txt','r') as fp:
			for line in fp.readlines():
				#file.write(line)
				lst = line.split("\n")
				parse = lst[0].split(", ")
				reg = parse[5]
		print(reg,Reg)
		Reg+= float(reg)

		os.system("rm out.txt")
	Reg/=50
	reg_list.append(Reg)

print(reg_list)
#plt.plot(dom,reg_list)
#plt.savefig('epsilon')
'''    
return ans
with open('outputDataT3_1.txt','w') as file:
	for inst in instance3:
		main_list = []
		for alg in algorithms:
			reg_list = []
			for T in horizon:
				Reg = 0
				for randseed in seed:
					s = "python3 bandit.py --instance " + inst + " --algorithm " + alg + " --randomSeed " + str(randseed) + " --epsilon " + str(eps) + " --horizon " + str(T) + " > out.txt"
					os.system(s)
					with open('out.txt','r') as fp:
						for line in fp.readlines():
							file.write(line)
							lst = line.split("\n")
							parse = lst[0].split(", ")
							reg = parse[5]
					print(reg)

					Reg += float(reg)

					

				print("\n")
				print("1 T done")		
				Avg_reg = Reg/50
				#Reg = 0
				reg_list.append(Avg_reg)

				
			print("1 alg done")	
			main_list.append(reg_list)

		for i in range(len(algorithms)):
			plt.plot(horizon,main_list[i],'-o')
			#name = "plot_" + str(i) + ".png"
			
			#plt.savefig(name)
			#plt.close()

		plt.legend(['epsilon-greedy','ucb','kl-ucb','thompson-sampling'])
		plt.title('Instance3 Regret vs Horizon')
		plt.xlabel('Horizon in Log scale')
		plt.ylabel('Regret')
		plt.xscale("log")
		name = "plot_instance_3"
		plt.savefig(name)
		plt.close()

'''