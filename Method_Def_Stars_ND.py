import numpy as np
import math
import time
import random
time1 = time.perf_counter()

bound = [-5,5]

num_args = 10
population_size = 20
k = 5

VF = 0.0001
VCH = 0.00001


#Sphere

# def Function(C):
#     sum = 0.
#     for i in range(num_args):
#         sum+=C[i]**2
#     return sum

#Растрігін
def Function(C):
    sum=0.
    for i in range(num_args):
        sum+= C[i]**2 - num_args * np.cos(2*np.pi * C[i])
    return 10*num_args + sum


def findDistance(pop):
    n=0
    sum=0.
    for i in range (len(pop)-1):
        for j in range(num_args-1):
            sum+= abs(pop[i].X[j]-pop[i+1].X[j])
            n+=1
    return sum / n


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

    def getTriangles(self):
        index = set(np.array(range(self.population_size), int))
        triangles =[]
        while(len(triangles)!=self.population_size):
            i,j,k = random.sample(index, 3)
            triangles.append([i,j,k])   
        return triangles
    
    def iterate(self):
        triangles = self.getTriangles()
        new_individsT = []
        new_individsQ = []
        new_individsU = []
        for item in triangles:
            #transfer
            i,j,k = item[0], item[1], item[2]
            individs = [self.population[i],self.population[j],self.population[k]]
            best = min(individs, key = lambda x: x.fit)
            index =individs.index(best)
            CentrMas = []
            for num in range(num_args):
                CentrMas.append(sum(individ.X[num] for individ in individs)/3)
                        
            new_individsT.append(Individ((1./(k-1)) * (k*best.X - CentrMas)))
            
            for m in range(3):
                if (m!= index):
                    new_individsT.append(Individ(1./k * ((k-1)*individs[m].X + new_individsT[0].X)))
            

            #rotate
            demK, demL = random.sample(set(np.array(range(num_args), int)),2)
            a = math.radians(np.random.randint(-360,360))
            for m in range(3):
                if(m!=index):
                    new_ind_X = individs[m].X 
                    new_ind_X[demK] = new_ind_X[demK]* np.cos(a) - new_ind_X[demL] * np.sin(a)
                    new_ind_X[demL] = new_ind_X[demK]* np.sin(a) + new_ind_X[demL] * np.cos(a)
                    new_individsQ.append(Individ(new_ind_X))
            
            #compress
            for m in range(3):
                if (m!= index):
                    new_individsU.append(Individ((k*best.X + individs[m].X)/(1+k)))
            
        return np.concatenate((new_individsT, new_individsQ,new_individsU))

        
class Individ(object):
    def __init__(self,number):
        self.X = np.array(number)
        self.checkBounds()
        self.fit=Function(self.X)                        
    def __str__(self):
        return 'X = {0} fit = {1}'.format(self.X, round(self.fit,5))
    def checkBounds(self):
        for i in range (num_args):            
            if self.X[i] < bound[0]:
                self.X[i] = self.X[i]+(bound[1]-bound[0])
            elif self.X[i] > bound[1]:
                self.X[i] = self.X[i]+(bound[0]-bound[1])
            else:
                self.X[i]= self.X[i]
        
           
def flatten(pop):
    array =[]
    for i in range(len(pop)):
        for j in range(len(pop[i])):
            array.append(pop[i][j])
    return array

N=0
popT = population(population_size, num_args)
popT.generate_population()

while( N<1000):
    Prev_fit = popT.fit
    new_Individs = popT.iterate()
    
    populations = np.concatenate((popT.population, new_Individs))
    
    populations = sorted(populations, key=lambda x: x.fit)[:population_size]
    best = populations[0]
    
    popT = population(population_size, num_args, populations)
    popT.fitness()
   
    # if(abs(Prev_fit-popT.fit)<VF):
    #     break

    if(findDistance(popT.population)<VCH):
        break
    print(N, " ",popT.fit)
    
    N+=1

print("Iterations = ", N)
print("General FIT = ",popT.fit)
print("Result:",popT.population[0])



time2 = time.perf_counter()

print(f"Time =  {time2 - time1:0.4f} seconds")