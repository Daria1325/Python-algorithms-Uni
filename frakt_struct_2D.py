import numpy as np
import random
import time
time1 = time.perf_counter()

bound = [-5,5]

num_args = 2
population_size = 30


VF = 0.000000001
VCH = 0.00000001

m=10
sigma = 0.2

#Eosom
def Function(X):
    rez = (-np.cos(X[0])*np.cos(X[1])*np.exp(-((X[0]-np.pi)**2 + (X[1]-np.pi)**2)))
    return rez

# My function
# def Function(C):
#     rez = (((C[0]**2 / np.sqrt(C[0]**2 + 1)) - (C[1]**2/ np.e**2))**2 * np.cos(C[0]**2 - C[1]))**3
#     return rez

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

    def calculate(self):
        squeres = self.getSquares()
        for i in range(len(squeres)):
            new_dots = self.find_new_dots(squeres[i])
            for item in new_dots:
                self.population=np.append(self.population,item)
        
    
    def getSquares(self):
        index = set(np.array(range(self.population_size), int))
        squares =[]
        i=0.
        while(len(squares)<self.population_size/2 and i<100):
            indexDots = random.sample(index, 2)
            dots = [self.population[indexDots[0]],self.population[indexDots[1]]]
            size = ((dots[0].X[0]-dots[1].X[0])**2 + (dots[0].X[1]-dots[1].X[1])**2)**0.5
            if size <0.5:
                size=0.5
            if np.random.random()<0.5:
                size=-size

            if dots[0].X[0]!=dots[1].X[0]:
                dots.append(Individ([dots[0].X[0]+size,dots[0].X[1]]))
                dots.append(Individ([dots[1].X[0]+size,dots[1].X[1]]))
                squares.append(dots)
            elif dots[0].X[1]!=dots[1].X[1]:
                dots.append(Individ([dots[0].X[0],dots[0].X[1]+size]))
                dots.append(Individ([dots[1].X[0],dots[1].X[1]+size]))
                squares.append(dots) 
            else:
                i+=1  
        return squares

    def find_new_dots(self,square):
        
        dot1 = [(square[0].X[0]+square[1].X[0])/2,(square[0].X[1]+square[1].X[1])/2] 
        dot2 = [(square[0].X[0]+square[2].X[0])/2,(square[0].X[1]+square[2].X[1])/2]
        dot3 = [(square[1].X[0]+square[3].X[0])/2,(square[1].X[1]+square[3].X[1])/2]
        dot4 = [(square[3].X[0]+square[2].X[0])/2,(square[3].X[1]+square[2].X[1])/2]
        square.append(Individ(dot1))
        square.append(Individ(dot2))
        square.append(Individ(dot3))
        square.append(Individ(dot4))
        square.append(Individ([(dot1[0]+dot4[0])/2,(dot2[1]+dot3[1])/2])) 

    
        new_individs = []
        for i in range(len(square)):
            for j in range(m):
                new_individs.append(Individ([square[i].X[0] +np.random.normal(0, sigma),square[i].X[1] +np.random.normal(0, sigma)]))
        return sorted(square, key=lambda x: x.fit)



    
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
            
        

pop = population(population_size, num_args)
pop.generate_population()
N=0


while( N<100):
    Prev_fit = pop.fit
    new_pop = []
    pop.calculate()
    
    populations = sorted(pop.population, key=lambda x: x.fit)
    i=0
    while(i<len(populations)-1):
        j=i+1
        while(j<len(populations)):
            if populations[i].fit==populations[j].fit:
                del populations[j]
            else:
                j+=1
        i+=1
    best = populations[0]
    
    pop = population(population_size, num_args, populations[:population_size])
    pop.fitness()
    N+=1

    # if(abs(Prev_fit-pop.fit)<VF):
    #     break

    # if(findDistance(pop.population)<VCH):
    #     break
    print("General FIT = ",pop.fit)


print("Iterations = ", N)
print("General FIT = ",pop.fit)
print("Result:",pop.population[0])



time2 = time.perf_counter()

print(f"Time =  {time2 - time1:0.4f} seconds")