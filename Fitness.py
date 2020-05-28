# Import Libraries
import numpy
import pandas
import random
import matplotlib.pyplot as plt

# Import Data
Table1 = pandas.read_excel('Data.xlsx', sheet_name='Distribution Centre', index_col=0, header=0)
Table2 = pandas.read_excel('Data.xlsx', sheet_name='Demand', index_col=0, header=0)
Table3 = pandas.read_excel('Data.xlsx', sheet_name='Vehicle Parameters', index_col=0, header=0)

# Position Coordinates
position_dc = Table1.values.tolist()
position_customers = pandas.DataFrame(Table2, columns=['X', 'Y']).values.tolist()
order = pandas.DataFrame(Table2, columns=['Order', 'ET', 'LT']).values.tolist()

# Fixed Parameters
price = 3000            # Rs. per quantity
C_dc = 1000             # DC fixed cost Rs.
C_k = 100               # truck fixed cost Rs.
k = 20                   # No. of trucks
L = 5                   # No. of DC
nPop = 60               # No. of customers
mileage_0 = 8           # km per Litre
mileage_m = 6           # km per Litre
load_capacity = 760     # kg for the truck
load_truck = 2000       # kg
load_max = load_truck + load_capacity
fuel = 70               # Rs. per Litre
speed = 0.4             # Km per min
t_unload = 15           # min
unit_mass = 10          # kg
mu_1 = 300              # Rs. per hour
mu_2 = 300              # Rs. per hour
delta_1 = 0.002         # Deterioration rate in transit
delta_2 = 0.003         # Deterioration rate when serving a customer
Q_dc = 200000           # kg for DC
RefCost_t = 15          # Rs. per hour
RefCost_u = 20          # Rs. per hour
Carbon_dc = 200         # Fixed Carbon Cost Rs.
C_fuel = 0.4            # Carbon Cost Rs. per kg


# Class
class getRoute:
    def __init__(self):
        self.S1 = substring1(nPop, K)
        self.S2 = substring2(K, open_dc)
        self.S3 = substring3(nPop)
        self.cost = 0
        self.carbon_qty = 0


# Variables
Z = numpy.zeros(L)              # Functioning DCs'
open_dc = []
for z in range(0, L):
    Z[z] = 1
    if Z[z] == 1:
        open_dc.append(z+1)

K = numpy.zeros(k)              # Functioning Trucks
for t in range(0, k):
    K[t] = 1


# Functions
# Mileage function
def mileage(load):
    consumption = mileage_0 + ((mileage_m - mileage_0) * (load - load_truck)) / (load_max - load_truck)
    return consumption


# Chromosomes
def substring1(customer, trucks):       # Substring 1
    string = []
    for i in range(0, customer):
        string.append(random.randint(1, sum(trucks)))
    return string


def substring2(trucks, dc):             # Substring 2
    string = []
    for i in range(0, int(sum(trucks))):
        string.append(random.choice(dc))
    return string


def substring3(customer):               # Substring 3
    string = []
    for i in range(0, customer):
        string.append(i + 1)
    random.shuffle(string)
    return string


# Single-point crossover Function
def single_crossover(ind1, ind2):
    ind1 = list(ind1)
    ind2 = list(ind2)
    r = random.randint(0, len(ind1))
    for i in range(r, len(ind1)):
        ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


# Double-point Crossover Function
def double_crossover(ind1, ind2):
    size = min(len(ind1), len(ind2))
    p1 = random.randint(1, size)
    p2 = random.randint(1, size - 1)
    if p2 >= p1:
        p2 += 1
    else:
        p1, p2 = p2, p1
    ind1[p1:p2], ind2[p1:p2] = ind2[p1:p2], ind1[p1:p2]
    return ind1, ind2


# Combined Crossover Function - Sequence Crossover + Partial matching Crossover
def uniform_crossover(ind1, ind2):
    firstCP = numpy.random.randint(0, nPop - 2)
    secondCP = numpy.random.randint(firstCP + 1, nPop - 1)
    ind1MC = ind1[firstCP:secondCP]
    ind2MC = ind2[firstCP:secondCP]
    temp_child1 = ind1[:firstCP] + ind2MC + ind1[secondCP:]
    temp_child2 = ind2[:firstCP] + ind1MC + ind2[secondCP:]
    child1 = list(recursion1(temp_child1, firstCP, secondCP, ind1MC, ind2MC))
    child2 = list(recursion2(temp_child2, firstCP, secondCP, ind1MC, ind2MC))
    return child1, child2


def recursion1(temp_child, firstCP, secondCP, ind1MC, ind2MC):
    relations = []
    for i in range(len(ind1MC)):
        relations.append([ind2MC[i], ind1MC[i]])

    child = numpy.array([0 for i in range(nPop)])
    for i, j in enumerate(temp_child[:firstCP]):
        c = 0
        for x in relations:
            if j == x[0]:
                child[i] = x[1]
                c = 1
                break
        if c == 0:
            child[i] = j
    j = 0
    for i in range(firstCP, secondCP):
        child[i] = ind2MC[j]
        j += 1

    for i, j in enumerate(temp_child[secondCP:]):
        c = 0
        for x in relations:
            if j == x[0]:
                child[i + secondCP] = x[1]
                c = 1
                break
        if c == 0:
            child[i + secondCP] = j
    child_unique = numpy.unique(child)
    if len(child) > len(child_unique):
        child = recursion1(child, firstCP, secondCP, ind1MC, ind2MC)
    return child


def recursion2(temp_child, firstCP, secondCP, ind1MC, ind2MC):
    relations = []
    for i in range(len(ind1MC)):
        relations.append([ind2MC[i], ind1MC[i]])

    child = numpy.array([0 for i in range(nPop)])
    for i, j in enumerate(temp_child[:firstCP]):
        c = 0
        for x in relations:
            if j == x[1]:
                child[i] = x[0]
                c = 1
                break
        if c == 0:
            child[i] = j
    j = 0
    for i in range(firstCP, secondCP):
        child[i] = ind1MC[j]
        j += 1

    for i, j in enumerate(temp_child[secondCP:]):
        c = 0
        for x in relations:
            if j == x[1]:
                child[i + secondCP] = x[0]
                c = 1
                break
        if c == 0:
            child[i + secondCP] = j
    child_unique = numpy.unique(child)
    if len(child) > len(child_unique):
        child = recursion2(child, firstCP, secondCP, ind1MC, ind2MC)
    return child


# Combined Cost Function
def cost_func(S1, S2, S3, Carbon):
    # This function comprises of six types of costs integrates namely;
    # Fixed Cost
    # Transportation Cost
    # Refrigeration Cost
    # Penalty Cost
    # Damage Cost
    # Carbon Emission Cost (given in order)
    C1 = 0; C61 = 0; C2 = 0; C31 = 0; C4 = 0; C51 = 0; C52 = 0; C62 = 0; Q6 = 0
    for i in range(0, L):
        fixed_truck = 0
        for j in range(0, k):
            fixed_truck = K[j]*C_k
        C1 += Z[i]*(C_dc + fixed_truck)
        C61 += Z[i]*Carbon_dc

    d = 0; count = 0
    for i in range(0, int(sum(K))):
        m = 0
        for j in range(0, len(S1)):
            if S1[j] == i + 1:
                position_current = S3[j] - 1
                m += order[position_current][0] * 10
        flag = 0; time_counter = 0
        C62 = m*C_fuel
        for j in range(0, len(S1)):
            if S1[j] == i + 1:
                flag += 1
                count += 1
                position_current = S3[j] - 1
                m -= order[position_current][0] * 10

                if flag == 1:
                    distance = ((position_dc[S2[i] - 1][0] - position_customers[position_current][0]) ** 2 + (
                            position_dc[S2[i] - 1][1] - position_customers[position_current][1]) ** 2) ** 0.5
                else:
                    distance = ((position_customers[position_current][0] - position_customers[position_last][0]) ** 2 + (
                            position_customers[position_current][1] - position_customers[position_last][1]) ** 2) ** 0.5

                d += distance
                position_last = position_current
                time_counter += distance / speed
                if time_counter / 60 < order[position_current][1]:  # early
                    C4 += mu_1 * ((time_counter / 60) - order[position_current][1])
                    time_counter += (order[position_current][1] - (time_counter / 60)) * 60
                elif time_counter / 60 > order[position_current][2]:  # late
                    C4 += mu_2 * ((time_counter / 60) - order[position_current][2])

                C2 += distance * fuel / mileage(m)
                C31 += (m * RefCost_t) * distance / (speed * 60)
                C51 += price * m * (1 - numpy.exp(-delta_1 * distance / speed))
                C52 += price * m * (1 - numpy.exp(-delta_2 * t_unload))
        Q6 += time_counter * Carbon/60
    C32 = (count * t_unload * RefCost_u)
    C3 = C31 + C32
    C5 = C51 + C52
    C6 = C61 + C62 + Q6
    Total = C1+C2+C3+C4+C5+C6
    return Total, Q6


# Genetic Algorithm for solving the LCLRP
def GeneticAlgo(MaxIt, Carbon_Q, reit):
    # Initializing Population
    pop = [getRoute() for p in range(nPop)]
    Child = [getRoute() for p in range(nPop)]
    for p in range(nPop):
        pop[p].cost, pop[p].carbon_qty = cost_func(pop[p].S1, pop[p].S2, pop[p].S3, Carbon_Q)

    for p in range(nPop-1):
        for q in range(0, nPop - p - 1):
            if pop[q].cost > pop[q + 1].cost:
                pop[q], pop[q + 1] = pop[q + 1], pop[q]

    global_best = getRoute()
    global_best.cost = numpy.inf

    # Main Loop
    log = []
    for it in range(MaxIt):
        for s in range(1, nPop, 2):
            Child[s - 1].S1, Child[s].S1 = single_crossover(pop[s - 1].S1, pop[s].S1)
            Child[s - 1].S2, Child[s].S2 = double_crossover(pop[s - 1].S2, pop[s].S2)
            Child[s - 1].S3, Child[s].S3 = uniform_crossover(pop[s - 1].S3, pop[s].S3)
        temp = pop
        for p in range(nPop):
            Child[p].cost, pop[p].carbon_qty = cost_func(Child[p].S1, Child[p].S2, Child[p].S3, Carbon_Q)
            temp.append(Child[p])

        for p in range(2*nPop - 1):
            # Last iter elements are already in place
            for q in range(0, 2*nPop - p - 1):
                if temp[q].cost > temp[q + 1].cost:
                    temp[q], temp[q + 1] = temp[q + 1], temp[q]
        pop = temp[0:nPop]
        if global_best.cost > pop[0].cost:
            global_best.cost = pop[0].cost
            global_best.S1 = pop[0].S1
            global_best.S2 = pop[0].S2
            global_best.S3 = pop[0].S3
            global_best.carbon_qty = pop[0].carbon_qty
        log.append(global_best.cost)
        print("Iteration " + str(it+1) + ": Minimum Cost = " + str(global_best.cost))
        if it == reit:
            reit += 50
            pop = [getRoute() for p in range(nPop)]
            for p in range(nPop):
                pop[p].cost, pop[p].carbon_qty = cost_func(pop[p].S1, pop[p].S2, pop[p].S3, Carbon_Q)
    return global_best, log


# Final Route
def RoutePlan(ind):
    Route_Plan = []
    for p in range(0, len(ind.S2)):
        route = []
        for q in range(0, len(ind.S1)):
            if ind.S1[q] == p + 1:
                route.append(ind.S3[q])
        route.insert(0, ind.S2[p]+nPop)
        Route_Plan.append(route)
    return Route_Plan


# Plot Function
def plot(string, string2, itr):
    x_dc = []; y_dc = []
    for i in range(len(position_dc)):
        x_dc.append(position_dc[i][0])
        y_dc.append(position_dc[i][1])
        plt.plot(x_dc[i], y_dc[i], 's', markersize=8, label='DC ' + str(i + 1))
    for i in range(0, len(string)):
        x_list = [position_dc[best.S2[i] - 1][0]]
        y_list = [position_dc[best.S2[i] - 1][1]]
        for j in range(1, len(string[i])):
            x_list.append(Table2.iloc[string[i][j] - 1]['X'])
            y_list.append(Table2.iloc[string[i][j] - 1]['Y'])
        x_list.append(position_dc[best.S2[i] - 1][0])
        y_list.append(position_dc[best.S2[i] - 1][1])

        plt.legend(bbox_to_anchor=(1, 1), loc='best', fontsize=5)
        plt.plot(x_list, y_list, '-o', markersize=3, linewidth=0.5, label='Route ' + str(i+1))
        plt.xlabel("X - coordinate")
        plt.ylabel("Y - coordinate")
        plt.title("LCLRP based on Genetic Algorithm")
    plt.savefig('Route '+str(itr))
    plt.figure()
    x_list = []; y_list = []
    for i in range(len(string2)):
        x_list.append(i)
        y_list.append(string2[i])
        plt.plot(x_list, y_list, '-o', markersize=1)
        plt.xlabel("Iteration")
        plt.ylabel("Best Total Cost")
        plt.title("Cost v/s Iteration")
    plt.savefig('Iteration '+str(itr))


# Main Loop
result = []
for C in range(11):
    best, logs = GeneticAlgo(5000, C, 50)
    routing = RoutePlan(best)
    print("Minimum Cost for C = " + str(C) + " is : " + str(best.cost) + " with plan : " + str(routing))
    result.append([C, best.cost, best.carbon_qty])
    plot(routing, logs, C)
df = pandas.DataFrame(result, columns=['C', 'Best Cost', 'Carbon Emission'])
print(df)
