import numpy as np
import random
import time
from scipy.spatial.distance import hamming

time1 = time.perf_counter()

VF=0.0001 #Критерій зупинки ВФ
VCH = 0.001 #Критерій зупинки ВЧ
num_arg = 2
dots=32
kol_b = 50 #к-ть батьків
kol_d = 7 #к-сть нащадків
kol = kol_d*kol_b+kol_b #к-сть всього
e = 0.001
sigma = 0.1
bounds = [-10,10]

class Individ:
    def __init__(self,num):
        self.C=[]
        for i in range(num_arg):
            self.C.append(num[i])
        self.fit=0                        
    def fitness(self):
       #self.fit=func_Sphere(self.C)
        #self.fit = func_Ros(self.C)
        #self.fit = func_Izum(self.C)
        #self.fit = func(self.C)
        self.fit = myFunc(self.C)
        
    def info(self):  
        self.fitness()
        #print("C = ",self.C,"fit = ",self.fit)                       
   
def func_Sphere(C):
    sum = 0.
    for i in range(num_arg):
        sum+=C[i]**2
    return(round(sum,6))                             
def func_Ros(C):
    sum = 0
    for i in range(num_arg-1):
        sum+=100*((C[i+1]-C[i]**2)**2) + (C[i]-1)**2
    return(round(sum,6))
def func_Izum(C):
    rez = (-np.cos(C[0])*np.cos(C[1])*np.exp(-((C[0]-np.pi)**2 + (C[1]-np.pi)**2)))
    return rez

def myFunc(C):
    rez = (((C[0]**2 / np.sqrt(C[0]**2 + 1))- (C[1]**2/ np.e**2))**2 * np.cos(C[0]**2 - C[1]))**3
    return rez

def func(C):
    one = C[1] * ((10*np.sin(C[0]/10.))/(50*np.cos(C[1]/10.)))
    two = C[0] * ((10*np.cos(C[1]/10.))/(50*np.sqrt(C[0]**2+7)))
    rez = one + two
    return rez

mas2 = []
for j in range (num_arg):
    mas=[]
    for i in range(kol_b):
        mas.append(random.uniform(bounds[0],bounds[1]))
        random.shuffle(mas)
    mas2.append(mas)
mas2 = np.array(mas2)
mas2 = np.transpose(mas2)

def findDistance(pop):
    sum = 0.
    for i in range (len(pop.B)-2):
        sum+= abs(pop.B[i].C[0]-pop.B[i+1].C[0])+ abs(pop.B[i].C[1]-pop.B[i+1].C[1])
    return sum / (len(pop.B)*2-2)



class population():
    def __init__(self):
        self.B = []
        for i in range(kol_b):
            p = Individ(mas2[i])
            self.B.append(p)
    def fitness(self):
        sum=0.
        for i in self.B:
            i.fitness()
            sum+=i.fit
        self.fit = round(sum/len(self.B),6)

    def info(self):
        for i in range(len(self.B)):
            self.B[i].fitness()
            self.B[i].info()
        self.fitness()
        #print("General FIT",self.fit)
    

def sort_fit(pop):
    pop.fitness()
    list=sorted(pop.B,key = lambda A: A.fit)
    for i in range(len(pop.B)):
        pop.B[i]=list[i]
    return pop
def norm(sigma):
    return np.random.normal(0, sigma**2)

def new_population(pop,sigma):
    pop1 = population()
    #pop1.B = pop.B[kol_b:]# L, M
    pop1.B = pop.B # L+M
    sort_fit(pop1)
    pop1.fitness()
    pop2 = population()
    for i in range(kol_b):
        pop2.B[i]=pop1.B[i]
    pop2.fitness()

    

    return pop2, sigma
def check_bounds(C):
    if C < bounds[0]:
        C = (bounds[1] - C)/C
    elif C > bounds[1]:
        C = (bounds[0]+C)/C

    return C

pop = population()
pop.info()
N=0
while(N<1000):
    Prev_fit = pop.fit #попередня популяція
    N+=1
    for i in range(kol_b):
        for j in range (kol_d):
            
            pop.B.append(Individ([check_bounds(round(x+norm(sigma),8)) for x in pop.B[i].C]))
        
    pop2, sigma = new_population(pop,sigma)
    #pop2.info()
    #print(pop2.fit)
    pop = population()
    pop = pop2
    # if(findDistance(pop2)<VCH):
    #     break
    if(abs(Prev_fit-pop2.fit)<VF):
    	break
print("X = ", pop2.B[0].C)
print("Y = ", pop2.B[0].fit)
print(N)

time2 = time.perf_counter()

print(f"Time =  {time2 - time1:0.4f} seconds")
