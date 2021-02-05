import numpy as np
import matplotlib.pyplot as plt

def Reliability(r, n):
    return 1 - (1-r)**n

# Initial values:
m = 5
w = np.array([7.0, 8.0, 8.0, 6.0, 9.0])
wv2 = np.array([1.0, 2.0, 3.0, 4.0, 2.0])
V = 110
C = 175
W = 200
alpha = np.array([2.33, 1.45, 0.541, 8.05, 1.95]) / 10**5
beta = np.array([1.5, 1.5, 1.5, 1.5, 1.5])

# Conditions:
def g1(r, n):
    return np.sum(wv2 * n**2) - V <= 0

def g2(r, n):
    return np.sum(alpha * (-1000/np.log(r))**beta * (n + np.exp(0.25*n))) - C <= 0

def g3(r, n):
    return np.sum(w*n*np.exp(0.25*n)) - W <= 0

def g(r, n):
    for i in range(len(r)):
        if(r[i]>1 or r[i]<0):
            return False
        if(n[i]<1):
            return False
    return True

# Chromosome
class Chromosome:
    def __init__(self):
        r = np.random.rand(m)
        n = np.random.randint(5, size=m) + 1
        self.x = np.concatenate((r, n), axis=0)
        self.sigma = np.random.rand(m+m)

# Fitness:
def fitness(p):
    r = p.x[0:5]
    n = p.x[5:10]
    if(g1(r, n) and g2(r, n) and g3(r, n) and g(r, n)):
        R = Reliability(r, n)
        return np.prod(R)
    return 0

# Mutation
def mutation(p:Chromosome, T, _T):
    x = p.x.copy()
    sigma = p.sigma.copy()
    _T *= np.random.rand()
    _N = np.random.normal(0, 1)
    for i in range(len(x)):
        T *= np.random.rand()
        N = np.random.normal(0, 1)
        if(i>=m):
            sigma[i] = sigma[i] * np.exp(_T*_N + 100*T*N)
        else:
            sigma[i] = sigma[i] * np.exp(_T*_N + T*N)
        if(sigma[i] < 0.0001):
            sigma[i] = 0.0001
        x[i] = x[i] + sigma[i] * N
        if(i>=m):
            x[i] = round(x[i])
    child = Chromosome()
    child.x = x.copy()
    child.sigma = sigma.copy()
    return child

# Ploting
def ploting(population, k, t):
    R = []
    N = []
    for i in range(len(population)):
        R += [population[i].x[t-1]]
        N += [population[i].x[t-1+m]]
    if(k % int(Iteration/10) == 0):
        plt.figure(str(t) + 'th subsystem')
        plt.subplot(2, 5, int(k/int(Iteration/10))+1)
        plt.xlim(0, 1)
        plt.ylim(0, 10)
        plt.plot(R, N, '.')
        plt.title(str(k) + 'th Iteration')
        plt.xlabel('r')
        plt.ylabel('n')

# Solving
Psize = 20 # Population size
Msize = 500 # Mutation size
Iteration = 50 # Iteration size
T = 1.0/(2*m)**(1/2)
_T = 1.0/(2*m**(1/2))**(1/2)
population = [] # Population


for i in range(Psize):
    population += [Chromosome()]

Y = [] # Best fitness
Ps = 0
for k in range(Iteration):
    for i in range(Msize):
        parent = population[int(np.random.randint(Psize))]
        child = mutation(parent, T, _T)
        if(fitness(child) > fitness(parent)):
            Ps += 1
        population += [child]

    population.sort(key=fitness, reverse=True)
    population = population[0 : Psize]

    if(Ps/Msize > 1/5):
        for i in range(len(population)):
            _C = np.random.rand(m+m) / 5 + 0.8
            population[i].sigma = population[i].sigma / _C
    elif(Ps/Msize < 1/5):
        for i in range(len(population)):
            _C = np.random.rand(m+m) / 5 + 0.8
            population[i].sigma = population[i].sigma * _C

    Y += [fitness(population[0])]

    ploting(population, k, 1)
    ploting(population, k, 2)
    ploting(population, k, 3)
    ploting(population, k, 4)
    ploting(population, k, 5)

plt.figure("Best Fitness")
plt.plot(Y, '-')
plt.xlabel("Iteration")
plt.ylabel("Fitness")

print("Reliability of each component:")
print(population[0].x[:m])
print("Number of components:")
print(population[0].x[m:])
print("Fitness:")
print(fitness(population[0]))

plt.show()
