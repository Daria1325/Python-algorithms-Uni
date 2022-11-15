import numpy as np
import random
import sympy
import time
from deap import tools
time1 = time.perf_counter()

C = [[3.4, 2.64],
     [5.4, 65.04],
     [6.7, 122.76],
     [8.2, 206.16],
     [9.12, 266.2176],
     [10.25, 349.25],
     [12.34, 529.7424],
     [21.43,1721.26],
     [23.76, 2133.11],
     [25.32, 2433.13]]

Mutation = 0.051    # Мутація
O_recombination = 0.3 # Одноточкова рекомбінація
T_recombination = 0.3 # Двоточкова рекомбінація
G_recombination = 0.1 # Генна рекомбінація
IS = 0.1              # Коефіцієнт IS переміщення
IS_length = [2,3]         # Довжина IS-елемента
RIS = 0.1             # Коефіцієнт RIS переміщення        
RIS_length =[2,3]          # Довжина RIS-елемента
M = 100              # Ступінь селекції
h = 6                # Довжина голови
n = 4                # К-сть генів
t = h*(n-1)+1         # Довжина гену
max_iter = 1000
pop_size = 30


T = ['a']
F = ['+','-','*','/']


class population(object):
    def __init__(self, population_size):
        self.population_size = population_size
        self.population = []
        self.fit = None
    
    def fitness(self):
        sum=0.
        for i in self.population:
            sum+=i.fit
        self.fit = sum/len(self.population)

    def generate_population(self,empty=None):
        self.population=[]
        for i in range(self.population_size):
            if empty:
                self.population.append(0)
            else:
                self.population.append(Individ())
                        
        
    def generate_probabilities_array(self):
        
        array = []
        probs=[]
        sum =0.0
        count=0.0
        for i in range(len(self.population)):
            if self.population[i].fit.is_infinite==False:
                count+=1
                sum+=self.population[i].fit
        
        avarege = abs(sum/count)
        for i in range(len(self.population)):
            if self.population[i].fit.is_infinite==False:
                norm = self.population[i].fit/avarege
                probs.append(norm)
            else:
                probs.append(0)

        max,min = probs[0],probs[0]
        for i in range(len(probs)):
            if  probs[i]!=0 and max< probs[i]:
                max= probs[i]
            if probs[i]!=0 and min> probs[i]:
                min= probs[i]
        if max!=min:
            for i in range(len(probs)):
                if probs[i]!=0:
                    probs[i] = int((probs[i] - min)/(max-min)*10)
        else:
            return [0]

        for i in range(len(probs)):
            if probs[i]!=0:
                array+=([i]*probs[i])
            
        return array
        
    
    def add_individ(self,individ):
        self.population.append(individ)

    def selectOne(self,population):
        max = sum([x.fit for x in population])
        selection_probs = [x.fit/max for x in population]
        return population[np.random.choice(len(population), p=selection_probs)]

    # def selection(self):
    #     best_individs =[]
    #     for i in range(len(self.population)):
    #         if self.population[i].fit.is_infinite==False:
    #             best_individs.append(self.population[i])
        
    #     best_individs = sorted(best_individs,key= lambda x:x.fit, reverse=True)
        
    #     best_individs = best_individs[:self.population_size]
    #     new_pop =[]
    #     for i in range(pop_size):
    #         new_pop.append(pop.selectOne(best_individs))
    #     self.generate_population(empty=True)
    #     for i in range(self.population_size):
    #         self.population[i]=new_pop[i]
    #     self.fitness()
        
        
    def selection(self):
        best_individs =[]
        for i in range(len(self.population)):
            if self.population[i].fit.is_infinite==False:
                best_individs.append(self.population[i])
        
        best_individs = sorted(best_individs,key= lambda x:x.fit, reverse=True)[:self.population_size]
        
               
        self.generate_population(empty=True)
        for i in range(self.population_size):
            self.population[i]=best_individs[i]
        array_with_probs=self.generate_probabilities_array()
        index_of_next_individs = random.choices(array_with_probs,k=self.population_size)
        next_individs =[]
        for index in index_of_next_individs:
            next_individs.append(self.population[index])
        self.generate_population(empty=True)
        for i in range(self.population_size):
            self.population[i]=next_individs[i]
        self.fitness()
        if array_with_probs[0]==0 and len(array_with_probs)==1:
            return 0
            

   
    def find_best(self):
        best =self.population[0]
        for i in self.population:
            if i.fit.is_infinite==False and best.fit<i.fit:
                    best.fit = i.fit

        return best

        


class Individ(object):
    def __init__(self,fit=None):
        self.Ch = self.generate_chrm()
        self.expression = None
        self.func = None
        if fit==None:
            self.fit=self.calculate_fit() 
        else:
            self.fit=0                       
        
    def copy(self,individ):
        for i in range(n):
            self.Ch[i] = individ.Ch[i]
    
    def generate_chrm(self):
        chrm = []
        for i in range(n):
            gene = random.choices(F, k=h)
            gene = gene +(['a'] * (t-h))
            chrm.append(gene)
        return(chrm)

    
    
    def get_expression(self):
        expression = ""
        for item in self.Ch:
            new =[]
            if item[0] in F:
                new =new + ['(',')',item[0],'(',')']
            else:
                new.append(item[0])

            i=1
            if len(new)!=1:
                flagToStop = True
                while(i<len(item)and flagToStop):
                    flagToStop = False
                    j=0
                    while(j<len(new)):
                        if (new[j]=='('and new[j+1]==')'):
                            flagToStop = True
                            if item[i] in F:
                                new =new[:j+1] + ['(',')',item[i],'(',')']+new[j+1:]
                                j+=5
                                i+=1
                            else:
                                new =new[:j+1] + [item[i]]+new[j+1:]
                                j+=1
                                i+=1
                        else:
                            j+=1             

            string_exp=""
            string_exp = string_exp.join(new)
            string_exp = string_exp.replace('(a)','a')
            string_exp = '+('+string_exp+')'
            expression+=string_exp
        self.expression = expression[1:]

    def calculate_function(self,a):
        return sympy.sympify(self.func).subs(dict(a=a))
    
    def get_function(self):
        self.func= sympy.simplify(self.expression)

    def calculate_fit(self):
        self.get_expression()
        self.get_function()
        sum=0.
        for i in C:
            calculated = self.calculate_function(i[0])
            sum+= M - np.abs(calculated - i[1])
        self.fit = sum
        return(sum)

    
    def mutation(self):
        if np.random.normal() < Mutation:
            new_individ = Individ(fit=False)
            new_individ.copy(self)
            index =np.random.randint(0,h-1,n)
            for i in range(n):
                new_individ.Ch[i][index[i]] = random.choices(F+T, k=1)[0]
            new_individ.fit = new_individ.calculate_fit()
            pop.add_individ(new_individ)
    def IS_moving(self):
        if np.random.normal() < IS:
            new_individ = Individ(fit=False)
            new_individ.copy(self)
            length = random.sample(IS_length,1)[0]
            num_chromosoms = random.sample(range(0, n), 2) # обираємо хромосоми які будуть змінюватись
            index =np.random.randint(1,h-1-length,2) # обираємо індекси елементів для переносу лише з голови в голову
            new_individ.Ch[num_chromosoms[1]][index[1]:index[1]+length] = new_individ.Ch[num_chromosoms[0]][index[0]:index[0]+length] 
            new_individ.fit = new_individ.calculate_fit()
            pop.add_individ(new_individ)
    
    def RIS_moving(self):
        if np.random.normal() < RIS:
            new_individ = Individ(fit=False)
            new_individ.copy(self)
            length = random.sample(RIS_length,1)[0]
            num_chromosom = np.random.randint(0,n) # обираємо хромосомн яка буде змінюватись
            index =np.random.randint(0,h-1-length) # обираємо індекси елементів для переносу лише з голови в голову
            new_individ.Ch[num_chromosom] = new_individ.Ch[num_chromosom][index:index+length] +new_individ.Ch[num_chromosom][:h-length]+new_individ.Ch[num_chromosom][h:]
            new_individ.fit = new_individ.calculate_fit()
            pop.add_individ(new_individ)
    def OD_recombination(self,individ):
        if np.random.normal() < O_recombination:
            new_individ1,new_individ2 = Individ(fit=False),Individ(fit=False)
            new_individ1.copy(self)
            new_individ2.copy(individ)
            index =np.random.randint(0,t-1)
            gene = np.random.randint(0,n)
            for i in range(n):
                if i>gene:
                    new_individ1.Ch[i],new_individ2.Ch[i] = new_individ2.Ch[i], new_individ1.Ch[i]
                if i==gene:
                    new_individ1.Ch[i][index:],new_individ2.Ch[i][index:] = new_individ2.Ch[i][index:], new_individ1.Ch[i][index:]
            new_individ1.fit = new_individ1.calculate_fit()
            pop.add_individ(new_individ1)
            new_individ2.fit = new_individ2.calculate_fit()
            pop.add_individ(new_individ2)
    def TD_recombination(self,individ):
        if np.random.normal() < T_recombination:
            new_individ1,new_individ2 = Individ(fit=False),Individ(fit=False)
            new_individ1.copy(self)
            new_individ2.copy(individ)

            index1,index2 =np.random.randint(0,t-1,size=2)
            gene1,gene2 = np.random.randint(0,n,size=2)
            if index1 >index2:
                index1,index2=index2,index1
            if gene1 >gene2:
                gene1,gene2=gene2,gene1
            
            for i in range(n):
                if i>gene1 and i<gene2:
                    new_individ1.Ch[i],new_individ2.Ch[i] = new_individ2.Ch[i], new_individ1.Ch[i]
                if i==gene1 and i!=gene2:
                    new_individ1.Ch[i][index1:],new_individ2.Ch[i][index1:] = new_individ2.Ch[i][index1:], new_individ1.Ch[i][index1:]
                if i==gene1 and i==gene2:
                    new_individ1.Ch[i][index1:index2],new_individ2.Ch[i][index1:index2] = new_individ2.Ch[i][index1:index2], new_individ1.Ch[i][index1:index2] 
                if i!=gene1 and i==gene2:
                    new_individ1.Ch[i][:index2],new_individ2.Ch[i][:index2] = new_individ2.Ch[i][:index2], new_individ1.Ch[i][:index2]
            new_individ1.fit = new_individ1.calculate_fit()
            pop.add_individ(new_individ1)
            new_individ2.fit = new_individ2.calculate_fit()
            pop.add_individ(new_individ2)

    def G_recombination(self,individ):
        if np.random.normal() < G_recombination:
            new_individ1,new_individ2 = Individ(fit=False),Individ(fit=False)
            new_individ1.copy(self)
            new_individ2.copy(individ)
            gene = np.random.randint(0,n)
            new_individ1.Ch[gene],new_individ2.Ch[gene] = new_individ2.Ch[gene], new_individ1.Ch[gene]
            new_individ1.fit = new_individ1.calculate_fit()
            pop.add_individ(new_individ1)
            new_individ2.fit = new_individ2.calculate_fit()
            pop.add_individ(new_individ2)
            
def predict(func,a):
    return sympy.sympify(func).subs(dict(a=a))


pop = population(pop_size)
pop.generate_population()
N=0
bests=[]


while N<max_iter:
    for i in range(pop_size):
        pop.population[i].mutation()
        pop.population[i].IS_moving()
        pop.population[i].RIS_moving()
        random_individs = np.random.randint(0,len(pop.population)-1,3)
        pop.population[i].OD_recombination(pop.population[random_individs[0]])
        pop.population[i].TD_recombination(pop.population[random_individs[1]])
        pop.population[i].G_recombination(pop.population[random_individs[2]])


    pop.selection()
    print("Pop ",N, "Fit = ",pop.fit)
    best =pop.find_best()  
    print("Best Fit: ",best.fit) 
    if best.fit>900:
        print("Найкращий вираз:",best.expression)   
        print("Функція:\n",best.func)
        print("Fit:\n",best.fit) 
    if best.fit>1000:
        break
    N+=1
    

best =pop.find_best()  
print("Найкращий вираз:",best.expression)   
print("Функція:\n",best.func)
print("Fit:\n",best.fit) 
print("Iterations:",N) 

print("Predicted")
print("F(10) =",predict(best.expression,10))
print("F(20) =",predict(best.expression,20))
print("F(30) =",predict(best.expression,30))


time2 = time.perf_counter()

print(f"Time =  {time2 - time1:0.4f} seconds")
