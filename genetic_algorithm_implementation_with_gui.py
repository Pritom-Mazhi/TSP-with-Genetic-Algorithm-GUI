import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt


'''We first create a City class that will allow us to create and handle our cities.
These are simply our (x, y) coordinates. Within the City class,
we add a distance calculation and a cleaner way to output the cities as coordinates with __repr__ .'''

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

'''We’ll also create a Fitness class. we’ll treat the fitness as the inverse of the route distance.
We want to minimize route distance, so a larger fitness score is better.
we need to start and end at the same place, so this extra calculation is accounted for the distance calculation.'''

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0

    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


'''Create the population
We now can make our initial population (first generation).
To do so, we need a way to create a function that produces routes that satisfy our conditions.
To create an individual, we randomly select the order in which we visit each city'''
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

'''This produces one individual, but we want a full population,
looping through the createRoute function until we have as many routes as we want for our population.'''
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

'''Determine fitness
To simulate our “survival of the fittest”, we can make use of Fitness to rank each individual
in the population. Output will be an ordered list with the route IDs and each associated fitness score.'''
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


'''Select the mating pool'''
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


'''Breed'''
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children


'''Mutate'''
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop



'''Repeat for next generation'''
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


'''Evolution in motion'''
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


'''Running the genetic algorithm-may change'''
cityList = []
for i in range(0,25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))



'''Plot the improvement'''
def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


'''GUI CODE BLOCK STARTS'''
from tkinter import *

#key down function
def click():
    popsize=textentry1.get() #this will collect the text from the text entry box
    elitesize=textentry2.get()
    mutationrate=textentry3.get()
    generation=textentry4.get()
    geneticAlgorithm(population=cityList, popSize=int(float(popsize)), eliteSize=int(float(elitesize)), mutationRate=int(float(mutationrate)), generations=int(float(generation)))
    geneticAlgorithmPlot(population=cityList, popSize=int(float(popsize)), eliteSize=int(float(elitesize)), mutationRate=int(float(mutationrate)), generations=int(float(generation)))

#main:
window = Tk()
window.title("My Genetic Algorithm")
window.configure(background="black")

#canvas
canvas = Canvas(window, width=1000, height=500)
#canvas.pack()

#My Photo
photo1 = PhotoImage(file="/home/pritom/Desktop/enam/venv/solo/guio.gif")
Label (window, image=photo1, bg="black") .grid(row=0, column=0, sticky=W)

#create lebel
Label (window, text="Enter the population size: ", bg="black", fg="white", font="none 12 bold") .grid(row=1, column=0, sticky=W)
Label (window, text="Enter the elite size: ", bg="black", fg="white", font="none 12 bold") .grid(row=2, column=0, sticky=W)
Label (window, text="Enter the mutation rate: ", bg="black", fg="white", font="none 12 bold") .grid(row=3, column=0, sticky=W)
Label (window, text="Enter the number of generations: ", bg="black", fg="white", font="none 12 bold") .grid(row=4, column=0, sticky=W)

#create a text entry box
textentry1 = Entry(window, width=20, bg="white")
textentry1.grid(row=1, column=1, sticky=W)
textentry2 = Entry(window, width=20, bg="white")
textentry2.grid(row=2, column=1, sticky=W)
textentry3 = Entry(window, width=20, bg="white")
textentry3.grid(row=3, column=1, sticky=W)
textentry4 = Entry(window, width=20, bg="white")
textentry4.grid(row=4, column=1, sticky=W)

#add a submit buttor
Button(window, text="Submit", width=6, command=click) .grid(row=5, column=1, sticky=W)
#mainloop always down below
window.mainloop()

'''GUI CODE BLOCK ENDS'''
