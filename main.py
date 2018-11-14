#import mesa
from model import KnowledgeModel
import matplotlib.pyplot as plt


# Number of agents in the model
popSize = 10

# Number of iterations
timeLength = 2

model = KnowledgeModel(popSize)
for i in range(timeLength):
	print("~~~~~ ROUND " + str(i) + " ~~~~~")
	print("")
	for agent in model.schedule.agents:
		print(agent.unique_id, agent.belief)
	model.step()

#agent_knowledge = [a.belief for a in model.schedule.agents]
#plt.hist(agent_knowledge)
#plt.show()
