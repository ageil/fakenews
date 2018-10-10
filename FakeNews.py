import numpy as np
import random
import matplotlib.pyplot as plt

## Set Parameters

#################################
########## Changeables ##########
#################################

#Population Size, N
N = 20 # 10, 20, 30, 40, 50, 100

#Number of individuals starting with false beliefs
F = 1 
R = 1

DATA = [0,0,0]

#Probability of receiving false information
falseProb = 0.75

#Probability of receiving correction
correctProb = 0.5

#Number of rounds
t = 100

# Period of False News Spreading
#firstPeriod = 5
#secondPeriod = 10

#################################
########## Initialize ###########
#################################

#Initialize population
#Individuals are "tagged" as 0 initially,
#  we will update this tag as they get information
#  code: 0 = no information received
#        1 = received false information
#        2 = received retraction
population = np.zeros(N)
#print "The Population is " + str(population)

for y in range(F):
    population[y] = 1

for y in range(F, F+R):
    population[y] = 2

#print "The new population is " + str(population)

# This will serve as a sanity check for the 
# Random number generator at the outset.
# We will count how many times each Individual is selected
selectionCounter = np.zeros(N)
#while notDone:
for y in range(0,t):

    #print "Round " + str(y)
    # Select individual
    A = random.randint(0,N-1)
    B = random.randint(0,N-1)
    ## Double check that this is a good way of 
    ## Getting unique individuals
    if A == B and A != N-1:
        B += 1
        #print "Shifted B Up"
    elif A == B:
        B -= 1
        #print "Shifted B Down"
    selectionCounter[A] += 1
    selectionCounter[B] += 1
    #print "On this round, A is "  + str(A)
    #print "On this round, B is "  + str(B)

    ## Spread beliefs
    if population[A] + population[B] == 1:
        population[A] = 1
        population[B] = 1
        #print "fake news has spread"
    elif population[A] + population[B] == 3:
        population[A] = 2
        population[B] = 2
        #print "truth has persevered"

    knowledgeCount = [0, 0, 0]
    for i in range(N):
        if population[i] == 0:
            knowledgeCount[0] += 1
        elif population[i] == 1:
            knowledgeCount[1] += 1
        elif population[i] == 2:
            knowledgeCount[2] += 1

    #print "Knowedge Count at End of Round " + str(y) + " is: "

    DATA = np.append(DATA,knowledgeCount)
    #print knowledgeCount

print DATA

#fig = plt.figure(figsize=(6, 3.2))

#ax = fig.add_subplot(111)
#ax.set_title('colorMap')
#plt.imshow(DATA)
#ax.set_aspect('equal')

#cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
#cax.get_xaxis().set_visible(False)
#cax.get_yaxis().set_visible(False)
#cax.patch.set_alpha(0)
#cax.set_frame_on(False)
#plt.colorbar(orientation='vertical')
#plt.show()

    ## Check if false beliefs are still Spreading
    #theSum = sum(population)
    #print "the Sum is: " + str(theSum)

    #if theSum == N or theSum == 2*N:
    #    notDone = False
    #    return notDone
    #print "We Finished on Round " + str(y)


    #print population
    #print "Selection Counter is: " + str(selectionCounter)
    #print sum(selectionCounter)
