import numpy as np
import math
import time
time1 = time.perf_counter()

bound = [-10,10]

num_args = 1
population_size = 10

sigma = (bound[1]-bound[0])/100.
# sigma = (bound[1]-bound[0])/10.
# sigma = (bound[1]-bound[0])/5.


VF = 0.01
VCH = 0.01


#Sphere
# def Function(X):
#     if( num_args ==2):
#         return X[0]**2 + X[1]**2
#     if(num_args==1):
#         return 2*(X[0]**2)

#Eosom
def Function(X):
    rez = (-np.cos(X[0])**2*np.exp(-2 *(X[0]-np.pi)**2 ))
    return rez

# def Function(C):
#     sum = 0
#     for i in range(num_args-1):
#         sum+=100*((C[i+1]-C[i]**2)**2) + (C[i]-1)**2
#     return sum

# def Function(C):
#     rez = (((C[0]**2 / np.sqrt(C[0]**2 + 1))- (C[1]**2/ np.e**2))**2 * np.cos(C[0]**2 - C[1]))**3
#     return rez

def findDistance(pop):
    sum = 0.
    for i in range (len(pop)-2):
        sum+= abs(pop[i].X[0]-pop[i+1].X[0])
    return sum / (len(pop)-2)


class population(object):
    def __init__(self, population_size, num_args, population = None):
        self.population_size = population_size
        self.num_args = num_args
        self.population = population
        self.fit = None
    
    def fitness(self):
        sum=0.
        for i in self.population:
            sum+=i.fit
        self.fit = sum/len(self.population)

    def generate_population(self):
        population = np.random.uniform(low=bound[0], high=bound[1], size=(self.population_size, self.num_args))
        
        self.population = np.array([Individ(p) for p in population])
        self.best = sorted(self.population, key=lambda x: x.fit)[0]
        self.fitness()


class Individ(object):
    def __init__(self,number):
        self.X = np.array(number)
        self.checkBounds()
        self.fit=Function(self.X)                        
    def __str__(self):
        return 'X = {0} fit = {1}'.format(self.X, round(self.fit,4))
    def checkBounds(self):
        for i in range (num_args):            
            if self.X[i] < bound[0]:
                self.X[i] = self.X[i]+(bound[1]-bound[0])
            elif self.X[i] > bound[1]:
                self.X[i] = self.X[i]+(bound[0]-bound[1])
            else:
                self.X[i]= round(self.X[i],3)          

N=0
popT = population(population_size, num_args)
popT.generate_population()



while( N<20):
    Prev_fit = popT.fit
    popZ = population(population_size, num_args)
    popZ.generate_population()

    for i in range(popT.population_size):
        popZ.population[i] = Individ(popT.population[i].X + np.random.normal(0, sigma**2))
    
    popS = population(population_size, num_args)
    popS.generate_population()
    
    for i in range(popT.population_size):
        indexes = np.random.randint(low =0, high =popT.population_size, size = 2)
        popS.population[i] = Individ((popT.population[indexes[0]].X + popT.population[indexes[1]].X)/2.)
    

    populations = np.concatenate((popT.population, popZ.population,popS.population))
    
    populations = sorted(populations, key=lambda x: x.fit)[:population_size]
    best = populations[0]
    
    popT = population(population_size, num_args, populations)
    popT.fitness()

    if(abs(Prev_fit-popT.fit)<VF):
        break

    # if(findDistance(popT.population)<VCH):
    #     break
    # print(N, " ",popT.fit)
    
    N+=1

for i in range(population_size):
    print(popT.population[i])
print("Iterations = ", N)
print("General FIT = ",popT.fit)
print("Result:",popT.population[0])



time2 = time.perf_counter()

print(f"Time =  {time2 - time1:0.4f} seconds")