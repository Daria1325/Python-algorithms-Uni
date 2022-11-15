import numpy as np
import random
import time
from scipy.spatial.distance import hamming

time1 = time.perf_counter()


num_arg = 2
step = 10
dots=2**step
kol = 20
e = round(10./(dots-1),8)
VF = 0.00001
# expect_x = np.pi
# expect_rez = -1
# expect_x = 0
expect_x = 1
expect_rez = 0

class Individ:
    def __init__(self,num):
        self.A = []
        self.C = []

        for i in range (num_arg):
            self.C.append(-1+num[i]*e)
            self.A.append(num[i])
        self.fit=0

    def fitness(self):
        #self.fit=func_Sphere(self.C)
        self.fit = func_Ros(self.C)
        #№self.fit = func_Izum(self.C)

    def info(self):
        self.fitness()
        print("C = ",self.C,"fit = ",self.fit)

def func_Sphere(C):
    sum = 0.
    for i in range(num_arg):
        sum+=C[i]**2
    return(round(sum,6))
def func_Ros(C):
    sum = 0
    for i in range(num_arg-1):
        sum+=100*((C[i+1]-C[i]**2)**2) + (C[i]-1)**2
    return(round(sum,7))
def func_Izum(C):
    rez = (-np.cos(C[0])*np.cos(C[1])*np.exp(-((C[0]-np.pi)**2 + (C[1]-np.pi)**2)))
    return rez



mas2 = []
for j in range (num_arg):
    mas=[]
    for i in range(dots):
        mas.append(i)
        random.shuffle(mas)
    mas2.append(mas)
mas2 = np.array(mas2)
mas2 = np.transpose(mas2)


class population():
    def __init__(self):
        self.B = []
        for i in range(kol):
            p = Individ(mas2[i])
            self.B.append(p)
    def fitness(self):
        sum=0.
        for i in self.B:
            i.fitness()
            sum+=i.fit
        self.fit = round(sum/len(self.B),7)

    def info(self):
        for i in range(kol):
            self.B[i].fitness()
            self.B[i].info()
        self.fitness()
        print("General FIT",self.fit)

#Елітний відбір
def new_population(pop):
    sort_(pop)
    
    pop2 = population()
    for i in range(kol):
        pop2.B[i]=pop.B[i]
    pop2.fitness()
    return pop2

def hasNotIndivid(individ, pop2, index):
    for i in range(index):
        if (pop2.B[i].A[0] == individ.A[0] and pop2.B[i].A[1]==individ.A[1]):
            return False
    return True

# Відбір з витісненням
def new_population1(pop):
    sort_(pop)
    pop2 = population()
    number = kol
    for i in range(number):
        if hasNotIndivid(pop.B[i],pop2, i):
            pop2.B[i]=pop.B[i]
        elif(number+1<len(pop.B)):
            number+=1
        else:
            break
    pop2.fitness()
    return pop2

#Звичайний відбір
def new_population2(pop):
    mas=[]
    for i in range(len(pop.B)):
        mas.append(i)
        random.shuffle(mas)
    pop2 = population()
    for i in range(kol):
        pop2.B[i]=pop.B[mas[i]]
    pop2.fitness()
    return pop2

def parent(pop):

    pop.fitness()
    SZ=pop.fit
    #Панміксія
    num1 = np.random.randint(0,kol-1)
    num2 = np.random.randint(0,kol-1)

    #Селективний
    # while(True):
    #     num1 = np.random.randint(0,kol-1)
    #     num2 = np.random.randint(0,kol-1)
    #     if(pop.B[num1].fit<SZ and pop.B[num2].fit<SZ):
    #         break

    #Інбридінг
    # num1 = np.random.randint(1,kol-1)
    # sort_(pop)
    # F1 = pop.B[num1].A[0]-pop.B[num1-1].A[0] + pop.B[num1].A[1]-pop.B[num1-1].A[1]
    # F2 = pop.B[num1].A[0]-pop.B[num1+1].A[0] + pop.B[num1].A[1]-pop.B[num1+1].A[1]
    # if(F1<=F2):
    #     num2=num1-1
    # else:
    #     num2=num1+1

    #Аутбридінг
    # num1 = np.random.randint(1,kol-1)
    # sort_A(pop)
    # F1=pop.B[num1].A[0]-pop.B[0].A[0]+pop.B[num1].A[1]-pop.B[0].A[1]
    # F2=pop.B[num1].A[0]-pop.B[kol-1].A[0]+pop.B[num1].A[1]-pop.B[kol-1].A[1]
    # if(F1>=F2):
    #     num2=0
    # else:
    #     num2=kol-1
    return (pop.B[num1], pop.B[num2])

def sort_(pop):
    pop.fitness()
    list=sorted(pop.B,key = lambda A: A.fit)
    for i in range(kol):
        pop.B[i]=list[i]
    return pop
def sort_A(pop):
    list=sorted(pop.B,key = lambda x: (x.A[0] + x.A[1])/2 )
    for i in range(kol):
        pop.B[i]=list[i]
    return pop
def best_child(pop):

        temp=sort_(pop)
        return(temp.B[0])

P =1 #ймовірність кросовер від 0,5 до 1
M = 0.05#ймовірність мутації
I = 0.01#ймовірність інверсії

def cross(parent):
    gen1=[]
    gen2=[]
    for i in range(num_arg):
        gen1.append(bin(parent[0].A[i])[2:])
        gen2.append(bin(parent[1].A[i])[2:])


    for i in range(num_arg):
        s1 = "0"*(step-len(gen1[i]))
        gen1[i]=s1+gen1[i]
        s2 = "0"*(step-len(gen2[i]))
        gen2[i]=s2+gen2[i]
    k = np.random.randint(0,step-1)
    for i in range (num_arg):
        gen1[i]=list(gen1[i])
        gen2[i]=list(gen2[i])

    for j in range(num_arg):
        for i in range(k,step):
            gen1[j][i], gen2[j][i] = gen2[j][i],gen1[j][i]


    if np.random.random()<0.5:
        return(gen1)
    else: return (gen2)

def cross2(parent):
    gen1=[]
    gen2=[]
    for i in range(num_arg):
        gen1.append(bin(parent[0].A[i])[2:])
        gen2.append(bin(parent[1].A[i])[2:])


    for i in range(num_arg):
        s1 = "0"*(step-len(gen1[i]))
        gen1[i]=s1+gen1[i]
        s2 = "0"*(step-len(gen2[i]))
        gen2[i]=s2+gen2[i]
    k = np.random.randint(0,step-5)
    k2=np.random.randint(step-5,step-1)
    for i in range (num_arg):
        gen1[i]=list(gen1[i])
        gen2[i]=list(gen2[i])

    for j in range(num_arg):
        for i in range(k,k2):
            gen1[j][i], gen2[j][i] = gen2[j][i],gen1[j][i]


    if np.random.random()<0.5:
        return(gen1)
    else: return (gen2)


def mutation(child):
    k = np.random.randint(0,step)
    for i in range(num_arg):
        if(child[i][k]=='0'):
            child[i][k]='1'
        else:
            child[i][k]='0'

def inversion(child):
    for i in range(num_arg):
        child[i] = child[i][::-1]

def create_child(child):
    new = Individ(mas2[0])
    for i in range(num_arg):
        new.A[i]=int(child[i],2)
        new.C[i]=(-1+new.A[i]*e)
    new.fit = new.fitness()
    return new

def hammingForPop(pop):
    sum=0
    gen1=[]
    gen2=[]
    for j in range(len(pop.B)):
        gen1.append(bin(pop.B[j].A[0])[2:])
        gen2.append(bin(pop.B[j].A[1])[2:])
    
        
        s1 = "0"*(step-len(gen1[j]))
        gen1[j]=s1+gen1[j]
        s2 = "0"*(step-len(gen2[j]))
        gen2[j]=s2+gen2[j]
    
        gen1[j]=list(gen1[j])
        gen2[j]=list(gen2[j])
        
    for j in range(len(gen1)-2):
        dis1 = hamming(gen1[j],gen1[j+1])
        dis2 = hamming(gen2[j],gen2[j+1])
        sum = sum + dis1 + dis2
    return sum/(len(gen1)*2)




pop1 = population()
pop1.info()

N=0
while (N<=10):
    sort_(pop1)
    N+=1
    pop2 = pop1
    for i in range(kol):
        parent1 = []
        temp=parent(pop1)
        parent1.append(temp[0])
        parent1.append(temp[1])

        if random.random()<P:
            child =cross2(parent1)
            if random.random()<M:
                mutation(child)
            for j in range(num_arg):
                child[j]=''.join(child[j])
            if random.random()<I:
                inversion(child)
            child = create_child(child)
            j=0
            flag=True
            for j in range(kol):
                if(child.A[0]==pop1.B[j].A[0] and child.A[1]==pop1.B[j].A[1]):
                    i-=1
                    flag=False
                    break

            if(flag):
                pop2.B.append(child)
        else:
            i-=1

    pop2 = new_population1(pop2)
    pop2.fitness()
    pop1.fitness()
    #ВФ Відстань між середніми значеннями цільової функції у сусідніх популяціях менше ніж
    # if(abs(pop2.fit - pop1.fit)<VF):
    #     temp = best_child(pop2)
    #     break
    print(hammingForPop(pop2))
    if (hammingForPop(pop2)<0.001):
         temp = best_child(pop2)
         break

    pop1 = pop2
    temp = best_child(pop2)
    sum=0

    
    # E =round(( abs(temp.C[0]-expect_x)**2 + abs(temp.C[1]-expect_x)**2),8)
    # if(E<=0.01):
    #     break
pop2.info()
print("Result")
print("X = ",temp.C)
print("Rez",temp.fit, "pohibka F",temp.fit-expect_rez)
print("epoch",N)

time2 = time.perf_counter()

print(f"Time =  {time2 - time1:0.4f} seconds")