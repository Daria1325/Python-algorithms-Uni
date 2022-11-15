import string
import numpy as np
import random
import sys


population_size = 10
n = 5           #к-сть кур'єрів піших
m = 5            #к-сть машин
p = 10          #к-сть пакунків
d = [[0.0,10,5.0,3.0,8.0,4.2],
     [10,0.0,4.0,2.0,4.7,7.0],
     [5.0,4.0,0.0,10,5.5,2.2],
     [3.0,2.0,10,0.0,8.9,11],
     [8.0,4.7,5.5,8.9,0.0,1.0],
     [4.2,7.0,2.2,11,1.1,0.0]] #матриця відстаней
vk = 4           #швидкість кур'єра
va = 60          #швидкість машини
zk = 90          #зарплата кур'єра
tsa = 15*50         #витрата пального ????



class Package(object):
    def __init__(self,address,weight,size):
        self.Address = address
        self.Weight = weight
        self.Size = size  # [length, width, hight]
        
    
    def __str__(self):
        return "Address: {}, Weight: {}, Size: {}\n".format(self.Address,self.Weight,self.Size)
    
    def detect_type(self):
        type = None
        if self.Weight>5:
            type = "Car"
        elif len(self.Size)==3 and (self.Size[0]>=0.3 or self.Size[1]>=0.3 or self.Size[2]>=0.3):
            type = "Car"
        else:
            type = "Human"
        return type


class population(object):
    def __init__(self, population_size,packages=None,population = None):
        self.population_size = population_size
        self.packages = packages
        self.fit = None
        self.best = None
        if population==None:
            self.population = []
        else:
            self.population = population

    def fitness(self):
        sum=0.
        for i in self.population:
            sum+=i.cost
        self.fit = sum/len(self.population)

    def generate_population(self,m,n):
        if self.doPackegesFit()==False:
            print("It is imposible to destribute all packeges as we don't have enough technical or human resourses")
            sys.exit()
            
            
        self.packages = sorted(self.packages, key=lambda x:x.Weight, reverse=True)
        self.population=[]
        for _ in range(self.population_size):
            self.population.append(Individ(m,n,clear=False,packages=self.packages))  
        self.fitness() 
        sortedPop=sorted(self.population, key=lambda x: x.cost)[:self.population_size]
        self.best = sortedPop[0]
    
    def doPackegesFit(self):
        weight=0.
        size=0.
        for pack in self.packages:
            weight+=pack.Weight
            if len(pack.Size)==3:
                size+= pack.Size[0]*pack.Size[1]*pack.Size[2]
        totalWeight = 200*m+5*n
        totalSize = 2*2*1*m
        if totalWeight<weight or totalSize<size:
            return False
        else:
            return True

    def mutate(self):
        for i in range(self.population_size):
            new =self.population[i].shuffle()
            if new !=None:
                self.population.append(new)

            new =self.population[i].exchange()
            if new !=None:
                self.population.append(new)

            new =self.population[i].returning()
            if new !=None:
                self.population.append(new)
               
        sortedPop=sorted(self.population, key=lambda x: x.cost)[:self.population_size] 
        self.find_best(sortedPop[0])
        self.update_pop(sortedPop)
              

    def find_best(self,popBest):
        if  self.best.cost> popBest.cost:
            self.best = popBest

    def update_pop(self,sortedPop):
        for i in range(self.population_size):
            self.population[i] = sortedPop[i]
        del self.population[self.population_size:]
        self.fitness()
    
    

class Individ(object):
    def __init__(self,m,n,clear,packages=None):

        self.Delivery = None
        self.cost = None
        self.generate(m,n, clear,packages)

    def __str__(self):
        text = ""
        for i in range(len(self.Delivery)):
            text+="\nRoute "+ str(i) +":\n"
            if i<m:
                text+= "Type: Car\n"
            else:
                text+= "Type Human\n"
            text+="Packages:\n"
            if self.Delivery[i]:
                for pack in self.Delivery[i]:
                    text+= str(pack)
                if i<m:
                    type="Car"
                else:
                    type="Human"
                text+= "Cost of route: {}\n".format(self.countOneRoute(self.Delivery[i],type))
            else:
                text+="Empty\n"
        text+="Cost of delivery: {}\n".format(self.cost)
        return text
    
    # def generate(self,m,n,clear,packages):
    #     deliveriesByCar = []
    #     deliveriesByHuman = []
    #     [deliveriesByCar.append([]) for _ in range(m)]
    #     [deliveriesByHuman.append([]) for _ in range(n)]
        
    #     if clear==False:
    #         for pack in packages:
    #             if pack.Type=="Car":
    #                 index=np.random.randint(0,m)
    #                 deliveriesByCar[index].append(pack)
    #             else:
    #                 #!!!! когда не хватаєт кур'єров
    #                 while self.check_available_postman(deliveriesByHuman):
    #                     index=np.random.randint(0,n)
    #                     if self.check_weight(deliveriesByHuman[index],pack):
    #                         deliveriesByHuman[index].append(pack)
    #                         break
    #         self.Delivery = deliveriesByCar+deliveriesByHuman
    #         self.count_cost()
    #     else:
    #         self.Delivery = deliveriesByCar+deliveriesByHuman
    def generate(self,m,n,clear,packages):
        deliveries = []
        [deliveries.append([]) for _ in range(m)]
        [deliveries.append([]) for _ in range(n)]
        
        if clear==False:
            for pack in packages:
                if pack.detect_type()=="Car":
                    while True:
                        index=np.random.randint(0,m)
                        
                        if self.check_weight("Car",route=deliveries[index],pack= pack) and self.check_size(route=deliveries[index],pack= pack):
                            deliveries[index].append(pack)
                            break
                else:
                    while True:
                        index=np.random.randint(0,n+m)
                        if index<m:
                            type="Car"
                            fitted=self.check_size(route=deliveries[index],pack= pack)#Check if has place in car
                        else:
                            type="Human"
                            fitted=True
                        if self.check_weight(type,route=deliveries[index],pack= pack) and fitted:
                            deliveries[index].append(pack)
                            break
            self.Delivery = deliveries
            self.count_cost()
        else:
            self.Delivery = deliveries

    


    def copy(self):
        newIndivid = Individ(m,n,clear=True)
        for i in range(len(self.Delivery)):
            newRoute=[]
            for item in self.Delivery[i]:
                newRoute.append(item)
            newIndivid.Delivery[i]= newRoute
        return newIndivid


    def count_cost(self):
        sumCar=0.
        for i,route in enumerate(self.Delivery):
            if i<m:
                type="Car"
            else:
                type = "Human"
            sumCar+= self.countOneRoute(route,type)
        self.cost = sumCar
        return

    def countOneRoute(self,route,type):
        sum=0.
        if route:
                dist=d[0][route[0].Address]
                for j in range(len(route)):
                    if j+1!= len(route):
                        dist += d[route[j].Address][route[j+1].Address]
                    else:
                        dist += d[route[j].Address][0]
                if type=="Car":
                    sum+= (dist / va) * zk + dist/100* tsa
                else:
                    sum+= (dist / vk) * zk
        return sum

    #Перемішування посилок в одному маршруті
    # def shuffle(self):
    #     newRout = []
    #     index = np.random.randint(0,len(self.Delivery))
    #     for i in range(len(self.Delivery[index])):
    #             newRout.append(self.Delivery[index][i])     
    #     np.random.shuffle(newRout)
        
    #     if self.countOneRoute(newRout)<self.countOneRoute(self.Delivery[index]):
    #         newInd=self.copy()
    #         newInd.Delivery[index]= newRout
    #         newInd.count_cost()
    #         return newInd
    #     else:
    #         return None
    def shuffle(self):
        newDelivery = []
        count=0
        for i,route in enumerate(self.Delivery):
            newRoute=[]
            for j in range(len(route)):
                newRoute.append(route[j])     
            np.random.shuffle(newRoute)

            if i<m:
                type="Car"
            else:
                type="Human"

            if self.countOneRoute(newRoute,type)<self.countOneRoute(route,type):
                newDelivery.append(newRoute)
            else:
                newDelivery.append(route)
                count+=1
        if count==len(self.Delivery):
            return None
        else:
            newInd=self.copy()
            for i in range(len(newDelivery)):
                newInd.Delivery[i]= newDelivery[i]
                newInd.count_cost()
                return newInd
                  
    
    #Обмін однією посилкою між доставками однакового типу
    # def exchange(self):
    #     if np.random.random()<0.5:
    #         indexRoute1, indexRoute2 = random.sample(range(0, m), 2)
    #     else:
    #         indexRoute1, indexRoute2 = random.sample(range(m, m+n), 2)
           

    #     if self.Delivery[indexRoute1] and self.Delivery[indexRoute2]:
    #         indexItem1 = np.random.randint(0,len(self.Delivery[indexRoute1]))
    #         indexItem2 = np.random.randint(0,len(self.Delivery[indexRoute2]))
    #     else:
    #         return None

    #     newRoute1 = self.copy()

    #     newRoute1.Delivery[indexRoute1].append(self.Delivery[indexRoute2][indexItem2])
    #     newRoute1.Delivery[indexRoute2].append(self.Delivery[indexRoute1][indexItem1])
        
    #     del newRoute1.Delivery[indexRoute1][indexItem1]
    #     del newRoute1.Delivery[indexRoute2][indexItem2]

    #     newRoute1.count_cost()
    #     #только людям машинам нет

    #     if self.check_weight(newRoute1.Delivery[indexRoute1])and self.check_weight(newRoute1.Delivery[indexRoute2]):
    #         return newRoute1
    #     else:
    #         return None
    def exchange(self):
        indexRoute1, indexRoute2 = random.sample(range(0, m+n), 2)
        if self.Delivery[indexRoute1] and self.Delivery[indexRoute2]:
            indexItem1 = np.random.randint(0,len(self.Delivery[indexRoute1]))
            indexItem2 = np.random.randint(0,len(self.Delivery[indexRoute2]))
        else:
            return None
        
        if not (indexRoute1<m and indexRoute2<m) or not (indexRoute1>=m and indexRoute2>=m): #Packs from different types of routes
            getType1Pack = self.Delivery[indexRoute1][indexItem1].detect_type()
            getType2Pack = self.Delivery[indexRoute2][indexItem2].detect_type()
            if getType1Pack!="Human" or getType2Pack!="Human": # one pack is Car the other Human
                if np.random.random()<0.5: # generate new routes of the same category
                    indexRoute1, indexRoute2 = random.sample(range(0, m), 2)
                else:
                    indexRoute1, indexRoute2 = random.sample(range(m, m+n), 2)

                if self.Delivery[indexRoute1] and self.Delivery[indexRoute2]:
                    indexItem1 = np.random.randint(0,len(self.Delivery[indexRoute1]))
                    indexItem2 = np.random.randint(0,len(self.Delivery[indexRoute2]))
                else:
                    return None

        newRoute = self.copy()

        newRoute.Delivery[indexRoute1].append(self.Delivery[indexRoute2][indexItem2])
        newRoute.Delivery[indexRoute2].append(self.Delivery[indexRoute1][indexItem1])
        
        del newRoute.Delivery[indexRoute1][indexItem1]
        del newRoute.Delivery[indexRoute2][indexItem2]

        newRoute.count_cost()

        flag =[True,True,True,True]
        if indexRoute1<m:
            type="Car"
            flag[2]=self.check_size(newRoute.Delivery[indexRoute1])
        else:
            type="Human"
        flag[0]=self.check_weight(type,newRoute.Delivery[indexRoute1])
        if indexRoute2<m:
            type="Car"
            flag[3]=self.check_size(newRoute.Delivery[indexRoute2])
        else:
            type="Human"
        flag[1]=self.check_weight(type,newRoute.Delivery[indexRoute2])
        
        if False not in flag:
            return newRoute
        else:
            return None
    

    #видалення посилки з однієї доставки й передача в іншу
    # def returning(self):
    #     if np.random.random()<0.5:
    #         indexRoute1 = np.random.randint(0,m)
    #         indexRoute2 = np.random.randint(0,m)
    #     else:
    #         indexRoute1 = np.random.randint(m,n+m)
    #         indexRoute2 = np.random.randint(m,n+m)


    #     newRoute1 = self.copy()

    #     if self.Delivery[indexRoute1]:
    #         indexItem = np.random.randint(0,len(self.Delivery[indexRoute1]))
    #         newRoute1.Delivery[indexRoute2].append(self.Delivery[indexRoute1][indexItem])
    #         del newRoute1.Delivery[indexRoute1][indexItem]
    #     elif self.Delivery[indexRoute1]:
    #         indexItem = np.random.randint(0,len(self.Delivery[indexRoute2]))
    #         newRoute1.Delivery[indexRoute1].append(self.Delivery[indexRoute2][indexItem])
    #         del newRoute1.Delivery[indexRoute2][indexItem]
    #     else:
    #         return None

    #     newRoute1.count_cost()

    #     if self.check_weight(newRoute1.Delivery[indexRoute1])and self.check_weight(newRoute1.Delivery[indexRoute2]):
    #         return newRoute1
    #     else:
    #         return None

    def returning(self):
        newRoute = self.copy()
        indexRoute1, indexRoute2 = random.sample(range(0, m+n), 2)
        if self.Delivery[indexRoute1] :
            indexItem = np.random.randint(0,len(self.Delivery[indexRoute1]))
            getTypePack = self.Delivery[indexRoute1][indexItem].detect_type()
            if (getTypePack=="Car" and indexRoute2<m) or getTypePack=="Human":
                newRoute.Delivery[indexRoute2].append(self.Delivery[indexRoute1][indexItem])
                del newRoute.Delivery[indexRoute1][indexItem]
        elif self.Delivery[indexRoute2]:
                indexItem = np.random.randint(0,len(self.Delivery[indexRoute2]))
                getTypePack = self.Delivery[indexRoute2][indexItem].detect_type()
                if (getTypePack=="Car" and indexRoute1<m) or getTypePack=="Human":
                    newRoute.Delivery[indexRoute1].append(self.Delivery[indexRoute2][indexItem])
                    del newRoute.Delivery[indexRoute2][indexItem]
        else:
            return None
        

        newRoute.count_cost()

        flag =[True,True,True,True]
        if indexRoute1<m:
            type="Car"
            flag[2]=self.check_size(newRoute.Delivery[indexRoute1])
        else:
            type="Human"
        flag[0]=self.check_weight(type,newRoute.Delivery[indexRoute1])
        if indexRoute2<m:
            type="Car"
            flag[3]=self.check_size(newRoute.Delivery[indexRoute2])
        else:
            type="Human"
        flag[1]=self.check_weight(type,newRoute.Delivery[indexRoute2])
        
        if False not in flag:
            return newRoute
        else:
            return None
        

    def check_weight(self,type, route=None, pack=None):
        weight =0.
        for item in route:
            weight+= item.Weight
        if pack==None:
            if type=="Car":
                return weight<=200
            else:
                return weight <=5.
        else:
            if type=="Car":
                return weight+pack.Weight<=200
            else:
                return weight+pack.Weight <=5.

    def check_size(self,route,pack=None):
        size =0.
        for item in route:
            if len(item.Size)==3:
                size+= item.Size[0]*item.Size[1]*item.Size[2]
        if pack==None:
            return size< 2*2*1
        else:
            if len(pack.Size)==3:
                size+= pack.Size[0]*pack.Size[1]*pack.Size[2]
            return size <2*2*1


            

def generate_package(num):
    packages=[]
    for i in range(num):
        address = i%5+1
        if i <num/2:
            weight = round(np.random.uniform(0.05,10),2)
        else:
            weight = round(np.random.uniform(50,150),2)
        if weight >5:
            size = np.random.uniform(0.3,1,3)
        else:
            if weight>0.2:
                size = np.random.uniform(0.1,0.3,3)
            else:
                size = np.random.uniform(0.1,0.3,2)
        packages.append(Package(address,weight,size))
    return packages


                



packages = generate_package(p)
pop = population(population_size,packages = packages)
pop.generate_population(m,n)

N=0
while N<100:
    pop.mutate()
    print("General Fit: ",pop.fit)
    print("Best: ", pop.best.cost)
    N+=1

print(pop.best)

