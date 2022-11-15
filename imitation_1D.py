import numpy as np
import math
import time
time1 = time.perf_counter()

bound = [-10,10]

num_args = 1
population_size = 30


VF = 0.01
VCH = 0.01

T_max = 100000000.
T_min = 100.

sigma = 0.5

#Eosom
def Function(X):
    rez = (-np.cos(X[0])**2*np.exp(-2 *(X[0]-np.pi)**2 ))
    return rez

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
        return 'X = {0} fit = {1}'.format(self.X, self.fit)
    def checkBounds(self):
        for i in range (num_args):            
            if self.X[i] < bound[0]:
                self.X[i] = self.X[i]+(bound[1]-bound[0])
            elif self.X[i] > bound[1]:
                self.X[i] = self.X[i]+(bound[0]-bound[1])
            else:
                self.X[i]= self.X[i]          
    def simulate_annealing(self, T):
        x_new = self.X + np.random.normal(0, sigma)
        ind_new = Individ(x_new)
        E = ind_new.fit - self.fit
        if E <=0:
            return ind_new
        else:
            r = np.random.uniform(low=0, high=1)
            if r < np.e**(-E/T):
                return ind_new
            else:
                return self
T = T_max
pop = population(population_size, num_args)
pop.generate_population()
I = 1.


while( T > T_min):
    Prev_fit = pop.fit
    new_pop = []
    for i in range(pop.population_size):
        new_individ = pop.population[i].simulate_annealing(T)
        if new_individ != pop.population[i]:
            pop.population = np.append(pop.population,new_individ)
    
    populations = sorted(pop.population, key=lambda x: x.fit)[:population_size]
    best = populations[0]
    
    pop = population(population_size, num_args, populations)
    pop.fitness()
    T = T/I
    I+=1    

    # if(abs(Prev_fit-pop.fit)<VF):
    #     break

    # if(findDistance(pop.population)<VCH):
    #     break


for i in range(population_size):
    print(pop.population[i])
print("Iterations = ", I)
print("Temperature = ", T)
print("General FIT = ",pop.fit)
print("Result:",pop.population[0])



time2 = time.perf_counter()

print(f"Time =  {time2 - time1:0.4f} seconds")