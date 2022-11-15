import numpy as np
import time


time1 = time.perf_counter()

num_args=1

F = 1
CR = 0.8
VF = 0.00001
VCH = 0.0001

def findDistance(pop):
    sum = 0.
    count = 0.
    for i in range (len(pop)-1):
        sum+= abs(pop[i][0]-pop[i+1][0])+ abs(pop[i][1]-pop[i+1][1])
        count+=2
    return sum / count

def findFitnessPop(pop):
    sum=0.
    for i in range(len(pop)):
        sum+=pop[i]
    return sum / len(pop)

def de(fobj, bounds, mut=F, crossp=CR, popsize=50, its=10000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        general_fit = findFitnessPop(fitness)
        
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        #print(general_fit)
        if abs(general_fit-findFitnessPop(fitness))<VF:
            print("Iteration", i)
            yield best, fitness[best_idx]
            break
        
        # distance = findDistance(pop)
        # if distance<VCH:
        #     print("Iteration", i)
        #     yield best, fitness[best_idx]
        #     break

        yield best, fitness[best_idx]


                           

#Function sphere
# fobj = lambda x: sum(x**2)
# bounds = [(-5, 5)] * 2
#End function sphere

# func
# def fobj(x):
#     one = x[1] * ((10*np.sin(x[0]/10.))/(50*np.cos(x[1]/10.)))
#     two = x[0] * ((10*np.cos(x[1]/10.))/(50*np.sqrt(x[0]**2+7)))
#     rez = one + two
#     return rez
# bounds = [(-100, 100)] * 2

#MyFunc
def fobj(C):
    rez = (((C[0]**2 / np.sqrt(C[0]**2 + 1))- (C[1]**2/ np.e**2))**2 * np.cos(C[0]**2 - C[1]))**3
    return rez
bounds = [(-10.,10.)]*2
#Function Rosenbrock
# def fobj(x):
#     sum = 0
#     for i in range(num_args-1):
#         sum+=100*((x[i+1]-x[i]**2)**2) + (x[i]-1)**2
#     return(round(sum,6))
# bounds = [(-20, 20)] * 2
#End function Rosenbrock

#Function Eosam
def fobj(X):
    rez = (-np.cos(X)**2*np.exp(-2 *(X-np.pi)**2 ))
    return rez
bounds = [(-10, 10)] 
#End function Eosam

it = list(de(fobj, bounds))
print(it[-1])



time2 = time.perf_counter()

print(f"Time =  {time2 - time1:0.4f} seconds")