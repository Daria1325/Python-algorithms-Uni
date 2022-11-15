import numpy as np
import math
import time
import random
time1 = time.perf_counter()

bound = [-5,5]

num_args = 2
population_size = 30
a = 1./4

VF = 0.00001
VCH = 0.001

#Sphere
# def Function(X):
#     if( num_args ==2):
#         return X[0]**2 + X[1]**2

#Rozenbrok
# def Function(C):
#     sum = 0
#     for i in range(num_args-1):
#         sum+=100*((C[i+1]-C[i]**2)**2) + (C[i]-1)**2
#     return sum

#Eosom
# def Function(X):
#     rez = (-np.cos(X[0])*np.cos(X[1])*np.exp(-((X[0]-np.pi)**2 + (X[1]-np.pi)**2)))
#     return rez

def Function(C):
    rez = (((C[0]**2 / np.sqrt(C[0]**2 + 1)) - (C[1]**2/ np.e**2))**2 * np.cos(C[0]**2 - C[1]))**3
    return rez





def findDistance(pop):
    sum = 0.
    n=0
    for i in range (len(pop)-1):
        sum+= abs(pop[i].X[0]-pop[i+1].X[0])+ abs(pop[i].X[1]-pop[i+1].X[1])
        n+=2
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

    def getRandomTriangles(self):
        index = set(np.array(range(self.population_size), int))
        triangles =[]
        n=0
        while(True and len(triangles)!=round(self.population_size/3)):
            i,j,k = random.sample(index, 3)
            x1, y1 = self.population[i].X[0],self.population[i].X[1]
            x2, y2 = self.population[j].X[0],self.population[j].X[1]
            x3, y3 = self.population[k].X[0],self.population[k].X[1]
            a = (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            n+=1
            if(a!=0):
                triangles.append([i,j,k])
                index.remove(i)
                index.remove(j)
                index.remove(k)
            if(len(index)<3):
                return triangles
            if(n>100):
                break
        return triangles
    
    def transfer(self, indexes):
        i,j,k = indexes[0], indexes[1], indexes[2]
        individs = [self.population[i],self.population[j],self.population[k]]
        CentrMas=[sum(items.X[0] for items in individs)/3,sum(items.X[1] for items in individs)/3]
        best = min(individs, key = lambda x: x.fit)
        index =individs.index(best)

        new_individs =[]
        new_individs.append(Individ([(1 + a)*best.X[0] - a*CentrMas[0],
                                     (1 + a)*best.X[1] - a*CentrMas[1]]))
        
        for i in range(3):
            if (i!= index):
                new_individs.append(Individ([(best.X[0]+ a*individs[i].X[0])/(1+a),
                                            (best.X[1]+ a*individs[i].X[1])/(1+a)]))
        return new_individs



        #sqweez
        # vecA = []
        # for i in range(3):
        #     if (i!= index):
        #         vecA.append([i,(best.X[0] - individs[i].X[0])/a,(best.X[1] - individs[i].X[1])/a])
        
        # individs[vecA[0][0]].X = [individs[vecA[0][0]].X[0]+vecA[0][1],
        #                           individs[vecA[0][0]].X[1]+vecA[0][2]]
                                  
        # individs[vecA[1][0]].X = [individs[vecA[1][0]].X[0]+vecA[1][1],
        #                           individs[vecA[1][0]].X[1]+vecA[1][2]]


        # #move
        # R = np.sqrt((CentrMas[0]-best.X[0])**2 + (CentrMas[1]-best.X[1])**2)
        # r = np.random.random() * R
        # vec = [best.X[0] - CentrMas[0],best.X[1] - CentrMas[1] ]
        # vec = [vec[0] * r, vec[1]*r]
        # population = []
        # for i in range(3):
        #     population = np.append(population, Individ([individs[i].X[0]+vec[0], individs[i].X[1]+vec[1]]))
        # return population

    def rotate(self, indexes):
        i,j,k = indexes[0], indexes[1], indexes[2]
        individs = [self.population[i],self.population[j],self.population[k]]
        best = min(individs, key = lambda x: x.fit)
        b = math.radians(np.random.randint(-360,360))

        population = []
        for i in range(3):
            x1 = (individs[i].X[0]- best.X[0])* np.cos(b) - (individs[i].X[1]-best.X[1])*np.sin(b)+ individs[i].X[0]
            x2 = (individs[i].X[0]- best.X[0])* np.sin(b) - (individs[i].X[1]-best.X[1])*np.cos(b)+ individs[i].X[1]
            if (x1!=best.X[0] and x2!= best.X[1]):
                population = np.append(population, Individ([x1, x2]))
        return population
    
    def rotateCentrum(self, indexes):
        i,j,k = indexes[0], indexes[1], indexes[2]
        individs = [self.population[i],self.population[j],self.population[k]]
        best = min(individs, key = lambda x: x.fit)
        CentrMas=[sum(items.X[0] for items in individs)/3,sum(items.X[1] for items in individs)/3]
        a = math.radians(np.random.randint(-360,360))

        population = []
        for i in range(3):
            x1 = (individs[i].X[0]- CentrMas[0])* np.cos(a) - (individs[i].X[1]-CentrMas[1])*np.sin(a)+ CentrMas[0]
            x2 = (individs[i].X[0]- CentrMas[0])* np.sin(a) - (individs[i].X[1]-CentrMas[1])*np.cos(a)+ CentrMas[1]
            population = np.append(population, Individ([x1, x2]))
        return population

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

    triangles = popT.getRandomTriangles()

    if(len(triangles)!=0):
        populationZ = []
        for i in range(len(triangles)):
            populationZ.append(popT.transfer(triangles[i]))
        populationZ = flatten(populationZ)
        popZ = population(len(populationZ),num_args,populationZ)  
        popZ.fitness()

        populationS = []
        for i in range(len(triangles)):
            populationS.append(popT.rotate(triangles[i]))
        populationS = flatten(populationS)
        popS = population(len(populationS),num_args,populationS)   
        popS.fitness()

        populationW = []
        for i in range(len(triangles)):
            populationW.append(popT.rotateCentrum(triangles[i]))
        populationW = flatten(populationW)
        popW = population(len(populationW),num_args,populationW)
        popW.fitness()
    else:
        popZ = population(population_size,num_args,[])
        popS = population(population_size,num_args,[])
        popW = population(population_size,num_args,[])
    

        
    
    populations = np.concatenate((popT.population, popZ.population,popS.population ,popW.population))
    
    # #populations = getUniqueValues(populations)
    
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

for i in range(population_size):
    print(popT.population[i])
print("Iterations = ", N)
print("General FIT = ",popT.fit)
print("Result:",popT.population[0])



time2 = time.perf_counter()

print(f"Time =  {time2 - time1:0.4f} seconds")