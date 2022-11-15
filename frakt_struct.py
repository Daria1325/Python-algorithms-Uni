import numpy as np
import math
import time
time1 = time.perf_counter()

bound = [-10,10]

population_size = 10


VF = 0.00001
VCH = 0.00001


m = 7
sigma = 0.2

#Eosom
def Function(X):
    rez = (-np.cos(X)**2*np.exp(-2 *(X-np.pi)**2 ))
    return rez

def findDistance(pop):
    sum = 0.
    for i in range (len(pop)-2):
        sum+= abs(pop[i].X-pop[i+1].X)
    return sum / (len(pop)-2)


class population(object):
    def __init__(self, population_size, population = None):
        self.population_size = population_size
        self.population = population
        self.fit = None
    
    def fitness(self):
        sum=0.
        for i in self.population:
            sum+=i.fit
        self.fit = sum/len(self.population)

    def generate_population(self):
        population = np.random.uniform(low=bound[0], high=bound[1], size=self.population_size)
        
        self.population = np.array([Individ(p) for p in population])
        self.best = sorted(self.population, key=lambda x: x.fit)[0]
        self.fitness()
    
  
class Individ(object):
    def __init__(self,number):
        self.X = number
        self.checkBounds()
        self.fit=Function(self.X)                        
    def __str__(self):
        return 'X = {0} fit = {1}'.format(self.X, self.fit)
    def checkBounds(self):
            if self.X < bound[0]:
                self.X = self.X+(bound[1]-bound[0])
            elif self.X > bound[1]:
                self.X = self.X+(bound[0]-bound[1])
            else:
                self.X= self.X         
    def generate_child(self):
        Sl, Sr,Ml,Mr=0.,0.,0.,0.
        
        for i in range(m):
            x_new = self.X +np.random.normal(0, sigma)
            if x_new < self.X:
                Sl += x_new
                Ml +=1
            else:
                Sr +=x_new 
                Mr +=1
        if Mr !=0 and Ml!=0:
            Xl = Individ(Sl/Ml)
            Xr = Individ(Sr/Mr)
            if Xl.fit < Xr.fit:
                return Xl
            else:
                return Xr
        elif Mr ==0:
            Xl = Individ(Sl/Ml)
            return Xl
        else:
            Xr = Individ(Sr/Mr)
            return Xr       

pop = population(population_size)
pop.generate_population()
N = 0


while( N<20):
    Prev_fit = pop.fit
    new_pop = []
    for i in range(pop.population_size):
        new_individ = pop.population[i].generate_child()
        pop.population = np.append(pop.population,new_individ)
    
    populations = sorted(pop.population, key=lambda x: x.fit)[:population_size]
    best = populations[0]
    
    pop = population(population_size,  populations)
    pop.fitness()
    N+=1   

    if(abs(Prev_fit-pop.fit)<VF):
        break

    if(findDistance(pop.population)<VCH):
        break


for i in range(population_size):
    print(pop.population[i])
print("Iterations = ", N)
print("General FIT = ",pop.fit)
print("Result:",pop.population[0])



time2 = time.perf_counter()

print(f"Time =  {time2 - time1:0.4f} seconds")