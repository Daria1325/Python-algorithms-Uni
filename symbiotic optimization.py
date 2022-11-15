import numpy as np
import time
time1 = time.perf_counter()

bf1=1
bf2=2

bound = (-10., 10.)
pop_size = 50 
num_args = 2 

VF=0.001 #Критерій зупинки ВФ
VCH = 0.001 #Критерій зупинки ВЧ


class SOS(object):
    def __init__(self, population_size, num_args):
        self.population_size = population_size
        self.num_args = num_args
        self.population = None
        self.best = None
        self.fit = None

    def generate_population(self):
        population = np.random.uniform(low=bound[0], high=bound[1], size=(self.population_size, self.num_args))
        self.population = np.array([Individ(p) for p in population])
        self.best = sorted(self.population, key=lambda x: x.fit)[0]
        self.fitness()

    def fitness(self):
        sum=0.
        for i in self.population:
            sum+=i.fit
        self.fit = sum/len(self.population)


    def mutualism(self, a_index):
        while(True):
            b_index = np.random.randint(0,self.population_size)
            if (b_index != a_index):
                break

        b = self.population[b_index]
        a = self.population[a_index]

        rand = np.random.random(2)
        mv = (a.C + b.C) / 2

        new_a = a.C + (rand * (self.best.C - (mv * bf1)))
        new_b = b.C + (rand * (self.best.C - (mv * bf2)))

        new_a = Individ([check_bounds(new_a[0]), check_bounds(new_a[1])])
        new_b = Individ([check_bounds(new_b[0]), check_bounds(new_b[1])])

        if new_a.fit < a.fit:
            self.population[a_index] = new_a
        if new_b.fit < b.fit:
            self.population[b_index] = new_b

        

    def commensalism(self, a_index):  
        while(True):
            b_index = np.random.randint(0,self.population_size)
            if (b_index != a_index):
                break  

        b = self.population[b_index]
        a = self.population[a_index]

        rand = np.random.uniform(low=-1, high=1, size=self.num_args)

        new_a = a.C + (rand * (self.best.C - b.C))
        new_a = Individ([check_bounds(new_a[0]),check_bounds(new_a[1])])

        if new_a.fit < a.fit:
            self.population[a_index] = new_a



    def parasitism(self, a_index):
        p_v = np.array(self.population[a_index].C)

        if np.random.random() < np.random.random():
            p_v[0] = np.random.uniform(low=bound[0], high=bound[1], size=1)
        else:
            p_v[1] = np.random.uniform(low=bound[0], high=bound[1], size=1)
        
        while(True):
            b_index = np.random.randint(0,self.population_size)
            if (b_index != a_index):
                break
        b = self.population[b_index]
        a = Individ(p_v)

        if (a.fit < b.fit):
            self.population[b_index] = a

class Individ(object):
    def __init__(self,number):
        self.C = np.array(number)
        self.fit=fitness_func(self.C)                        
    def __str__(self):
        return 'C = {0} fit = {1}'.format(self.C, self.fit)
    

def findDistance(pop):
    sum = 0.
    for i in range (len(pop)-2):
        sum+= abs(pop[i].C[0]-pop[i+1].C[0])+ abs(pop[i].C[1]-pop[i+1].C[1])
    return sum / (len(pop)*2-2)

def check_bounds(a):
    if a < bound[0]:
        a = (bound[1] - a)/a
    elif a > bound[1]:
        a = (bound[0]+a)/a
    return a

def fitness_func(C):
    # Sphere model 
    # sum = 0.
    # for i in range(num_args):
    #     sum+=C[i]**2
    # return(round(sum,6)) 

    # Rosenbrock
    # sum = 0
    # for i in range(num_args-1):
    #     sum+=100*((C[i+1]-C[i]**2)**2) + (C[i]-1)**2
    # return(round(sum,6))

    #Easom
    # rez = (-np.cos(C[0])*np.cos(C[1])*np.exp(-((C[0]-np.pi)**2 + (C[1]-np.pi)**2)))
    # return rez

    #Func
    # one = C[1] * ((10*np.sin(C[0]/10.))/(50*np.sqrt(C[1]**2+7)))
    # two = C[0] * ((10*np.cos(C[1]/10.))/(50*np.sqrt(C[0]**2+7)))
    # rez = one + two
    # return rez

    #myFunc
    rez = (((C[0]**2 / np.sqrt(C[0]**2 + 1))- (C[1]**2/ np.e**2))**2 * np.cos(C[0]**2 - C[1]))**3
    return rez
    




sos = SOS(pop_size, num_args)
sos.generate_population()

N=0

while(N<1000):
    Prev_fit = sos.fit
    for i, val in enumerate(sos.population):
        sos.mutualism(i)
        sos.commensalism(i)
        sos.parasitism(i)
        sos.best = sorted(sos.population, key=lambda x: x.fit)[0]
        sos.fitness()
            
    # if(findDistance(sos.population)<VCH):
    #     break

    if(abs(Prev_fit-sos.fit)<VF):
        break
    N+=1

print("Result:")
print("Iterations:", N)
print(sos.best)
time2 = time.perf_counter()

print(f"Time =  {time2 - time1:0.4f} seconds")