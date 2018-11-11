#import mesa
import random
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation

# Number of agents in the model
popSize = 10

# Number of iterations
timeLength = 2

class PopAgent(Agent):
	"""An Agent with some initial knowledge."""
	def __init__(self, unique_id, model):
		super().__init__(unique_id, model)
		#Set every agent to no information at the outset
		self.knowledge = 0

	def step(self):
		# During each step, we pick an agent to interact with self.
		other_agent = random.choice(self.model.schedule.agents)
		# Check to make sure self and other are unique
		# If not, sample again (until unique agent found)
		while other_agent.unique_id == self.unique_id:
			other_agent = random.choice(self.model.schedule.agents)
		# Sanity check, to ensure the agents are unique by this point.
		if other_agent.unique_id == self.unique_id:
			assert False, "Something is awry!"
		print("Agent-A is: " + str(self.unique_id))
		print("Agent-A knows: " + str(self.knowledge))
		print("Agent-B is: " + str(other_agent.unique_id))
		print("Agent-B knows: " + str(other_agent.knowledge))
		"""The way the knowledge flows is as follows: 
			If the sum of the agents' knowledge is 0,
				Neither agent has info.
				DO NOTHING.
			If the sum of the agents' knowledge is 1,
				One agent has no info and one agent has false info
				UPDATE both agents to false info
			If the sum of the agents' knowledge is 2,
				either both agents have false info or,
				one agent has no info and one agent has true info
				DO NOTHING
			If the sum of the agents' knowledge is 3,
				One agent has true info, one agent has false info
				UPDATE both agents to true info
			If the sum of the agents' knowledge is 4,
				Both agents have true info,
				DO NOTHING"""
		if self.knowledge + other_agent.knowledge == 1:
			self.knowledge = 1
			other_agent.knowledge = 1
			print("Fake news has spread")
			print("Agent-A is: " + str(self.unique_id))
			print("Agent-A knows: " + str(self.knowledge))
			print("Agent-B is: " + str(other_agent.unique_id))
			print("Agent-B knows: " + str(other_agent.knowledge))
		elif self.knowledge + other_agent.knowledge == 3:
			self.knowledge = 2
			other_agent.knowledge = 2
			print("agents have seen reason!")
			print("Agent-A is: " + str(self.unique_id))
			print("Agent-A knows: " + str(self.knowledge))
			print("Agent-B is: " + str(other_agent.unique_id))
			print("Agent-B knows: " + str(other_agent.knowledge))
		print("")

class KnowledgeModel(Model):
	"""A model with some number of agents."""
	def __init__(self, N):
		self.num_agents = N
		self.schedule = RandomActivation(self)
		#Create agents
		for i in range(self.num_agents):
			a = PopAgent(i, self)
			if i == 0:
				#Give Agent 0 false information
				a.knowledge = 1
			if i == 1:
				#Give Agent 1 true information
				a.knowledge = 2
			self.schedule.add(a)
	
	def step(self):
		"""Advance the model by one step."""
		self.schedule.step()

model = KnowledgeModel(popSize)
for i in range(timeLength):
	print("~~~~~ ROUND " + str(i) + " ~~~~~")
	print("")
	model.step()

#agent_knowledge = [a.knowledge for a in model.schedule.agents]
#plt.hist(agent_knowledge)
#plt.show()
