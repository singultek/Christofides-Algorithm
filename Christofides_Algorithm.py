
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt 
from collections import defaultdict 
import tsplib95
import operator
from copy import copy
import tkinter as Tk
from tkinter import filedialog


import gurobipy as gp
from   gurobipy import GRB
import itertools


#%% Getting TSP input manually from user
##############
    

#Function that, given a matrix A, returns its symmetric matrix
def sym_matrix(A):
    
    #For each non-zero element of the matrix, we assign its value to the 
    #corresponding symmetric element
    
    for i in range(0, np.shape(A)[0]):
        for j in range(i, np.shape(A)[1]):
            if A[i][j] != 0:
                A[j][i] = A[i][j]
                
    return (A)
    


################################

#Input from user


def user_input():

    global A
    global nodes
    global Weighted_ARC
    global Weighted_ARC_Sorted
    global ARC
    global input_nodes
    
    #Number of nodes to insert in the graph
    nodes = int(input("Number of nodes:"))
    
    input_nodes = nodes

    #Maximum number of arcs to insert
    numarcs = int(nodes*(nodes-1)/2)
    
    ARC = []
    
    arc = np.zeros((numarcs,2))

    print(f"Please insert s keyword to stop writing nodes\n")
    print(f"User should give positive integers for arcs\n")
    print(f"Number of given arcs should be at least {nodes-1} and at most {numarcs}\n")


    for i in range(0, numarcs) :
        
        beginning = input(f'Insert node of beginning of the arc {i+1}: ')
        arrival = input(f'Insert node of arrival of the arc {i+1}: ')
        
        if beginning == 's' or beginning == 'S' or arrival == 's' or arrival == 'S':
            break
        
        else:
            arc[i][0] = int(beginning)
            arc[i][1] = int(arrival)
            arc[i].sort()
            
            #Stop without writing all the arcs    
            if arc[i][0] > 0 and arc[i][1] > 0:      
                ARC.append(tuple(arc[i]))
            else:
                break

    #Definition of adjacency matrix
    A = np.zeros((nodes,nodes))
    
    Weighted_ARC = []
    Weighted_ARC_Sorted = []

    for i in ARC:

        Weighted_ARC.append([i[0], i[1], int(input("Give weight to the arc :" ))])
    
        if int(i[0]) and (i[1]) != 0 : 
            A[int(i[0])-1] [int(i[1])-1] = 1

    sym_matrix(A)
    print(A)
    ARC = [i for i in ARC if i[0] and i[1] != 0]
    print(ARC)
    
    #Ordering of the arcs given by the weight
    Weighted_ARC_Sorted = Weighted_ARC.copy()
    Weighted_ARC_Sorted.sort(key = operator.itemgetter(2))   
    print(Weighted_ARC)
    print(Weighted_ARC_Sorted)
    
    
    return 


#%% Getting TSP input from tsp file format
##########################

#Input from TSPLib


def tsp_input():
    
    global problem
    global ARC
    global A
    global Weighted_ARC
    global Weighted_ARC_Sorted
    global input_nodes
    
    #Uploading of input data from following directory

    roots = Tk.Tk()
    roots.withdraw() 
    roots.filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("tsp files","*.tsp"),("all files","*.*")))

    problem = tsplib95.load(roots.filename)
    nodes = list(problem.get_nodes())
    input_nodes = len(nodes)
    
    coords = []
    ARC = []
    
    for i in nodes:
        coords.append(problem.node_coords[i])
    
    #Initialization of adjacency matrix
    A = np.ones((len(nodes),len(nodes)))
    A = np.triu(A)
    for i in nodes:
        A[i-1][i-1] = 0

    Weighted_ARC = []
    Weighted_ARC_Sorted = []    
    
    for i in nodes:
        for j in nodes:
            if A[i-1][j-1] == 1:
                #Weight of each arc is the distance between the two extremal nodes
                x = ((coords[j-1][0]-coords[i-1][0])**2 +(coords[j-1][1] - coords[i-1][1])**2)**(1/2)
                Weighted_ARC.append([i, j, x])
                ARC.append(tuple([i,j]))

    #Ordering of the arcs given by the weight 
    Weighted_ARC_Sorted = Weighted_ARC.copy()
    Weighted_ARC_Sorted .sort(key = operator.itemgetter(2))           

    return  

#%% Drawing Graph
##################################


#Function that draws the graph with the input of edges

def simple_print_graph(G):

    nx.draw(G, with_labels = True)

    plt.show(G)

    return(G)

#%% Converting input list to adjacency matrix
#########################


#Function that, converts array of adjacency matrix into dictionary data type and 
#adds all neighbors nodes of the each corresponding node
def convert(a): 
    
    global adjList
    adjList = defaultdict(list)
    for i in range(len(a)):             
        for j in range(len(a[i])):      
            if a[i][j]== 1: 
                adjList[i + 1].append(j+1) 
                
    return adjList


#%% MST by using Kruskal
#########################################


#Determine the root of the subset of nodes to which node x belongs to,
#by following the preceding nodes of x until reaching a self loop 
def check_root(preceding_node_list, x): 
    
    if preceding_node_list[x] == x: 
        return x 
    return check_root(preceding_node_list, preceding_node_list[x]) 


#Function that adds an edge between the root nodes of the two trees
#containing x and y, in order to combine them
def unique_tree(preceding_node_list, depth, x, y): 
    
    #Find the two roots of trees containing x and y
    root_x = check_root(preceding_node_list, x) 
    root_y = check_root(preceding_node_list, y) 
    
    #If the depths of the two trees are equal, one root node becomes 
    #the preceding node of the other.
    #We increment the depth
    if depth[root_x] == depth[root_y]: 
        preceding_node_list[root_y] = root_x 
        depth[root_x] += 1
        
    
    #Otherwise, make the root node of the tree with smaller depth be the 
    #preceding node of the other root
    elif depth[root_x] > depth[root_y]: 
        preceding_node_list[root_y] = root_x 
    else: 
        preceding_node_list[root_x] = root_y 

    

#Function that constructs the Minimum Spanning Tree starting from the weighted 
#edges sorted with respect to the weight (length) using the functions 
#check_root and unique_tree 

def Kruskal():

    global MST_ARC
    MST_ARC = []    
    
    #Sort all the edges in non-decreasing order with respect to the weight 
    Weighted_ARC_Sorted = Weighted_ARC.copy()
    Weighted_ARC_Sorted.sort(key = operator.itemgetter(2)) 
    
    #Index variable, for sorted edges
    i = 0 
    #Index variable, for edges in MST
    e = 0 
    
    #List for storing the preceding nodes
    preceding_nodes = [] 
    #List for storing the depths of the sub-trees we are creating
    depth = [] 
    
    for j in range(input_nodes): 
            preceding_nodes.append(j) 
            depth.append(0) 
     
    
    #Number of arcs in MST is the number of nodes - 1 
    while e < input_nodes - 1:  
        
        m = int(Weighted_ARC_Sorted[i][0]) - 1
        n = int(Weighted_ARC_Sorted[i][1]) - 1
        
        #We have to check if, including the edge, we create a cycle.
        #We determine the root nodes of the subsets (trees) containing 
        #p1 and p2
        p1 = check_root(preceding_nodes, m)
        p2 = check_root(preceding_nodes, n)

        #If the the root nodes are different, we can add the edge to the MST
        #without creating a cycle and we increment the variable related to
        #the number of edges in MST
        if p1 != p2:
            MST_ARC.append((m + 1,n + 1))
            e = e + 1
        
        #Join of the two trees containing nodes x and y to create a new one
        unique_tree(preceding_nodes, depth, p1, p2)
        i = i + 1
        
    print(f"{MST_ARC}\n")
    print("End of Kruskal algorithm\n")
    
    #Print of the MST obtained with Kruskal algorithm
    H = nx.Graph() 
    H.add_edges_from(MST_ARC)
    simple_print_graph(H)

    return


#%% MST by using Prim
##########################


#Function that checks whether the key value is minimum or not

def check_min(s):
    global min_key
    
    #If weight is smaller than key value (initially infinity), replace the values
    #Then get and store the min value in min_key
    
    for i in key:
        if weight_adjacent_matrix[s-1][i-1] != 0 and mst_touched[i] != True:
            if key[i] > weight_adjacent_matrix[s-1][i-1]:
                key[i] = weight_adjacent_matrix[s-1][i-1]
                
        min_key = min(key, key=key.get)

    #Checking weight value of min_key in the weight adjacent matrix
    #When the the weight is found, we add corresponding edge into MST_ARC
    
    for i in range(len(weight_adjacent_matrix)):
        for j in range(len(weight_adjacent_matrix[i])):
            if weight_adjacent_matrix[i][j] == key[min_key] and min_key == i + 1 or weight_adjacent_matrix[j][i] == key[min_key] and min_key == j + 1:
                MST_ARC.append((i+1,j+1))
        
    #Adding min_key into mst_tour since we visit the node now 
    
    mst_tour.append(min_key)
    print(f"Selected node is : {min_key}\n")
    
    return min_key

#Function that updates the visited nodes values in key and mst_touched dictionaries

def replace_min_key():
    
    key[min_key] = inf
    mst_touched[min_key] = True

    return key


def Prim():

    # Initialization 
    #Creating a list with all nodes
    
    global nodes
    nodes = []
    for i in range(1,input_nodes+1):
        nodes.append(i)
        
    #Creating adjacent matrix by using weight of edges
    
    global weight_adjacent_matrix 
    weight_adjacent_matrix = np.zeros((len(nodes),len(nodes)))
    for i in Weighted_ARC:
        weight_adjacent_matrix[int(i[0]-1)][int(i[1]-1)] = i[2]
        weight_adjacent_matrix[int(i[1]-1)][int(i[0]-1)] = i[2]
        
    print(f"Matrix form of weight arc : {weight_adjacent_matrix}\n")
    
    #Dictionary of key is created and all values are initially infinity
    
    global key
    key = {}
    global inf
    inf = 999999
    global initial_node
    initial_node = 1
    min_key = initial_node
    
    for i in nodes:
        key[i] = inf
    
    #Creating a dictionary for storing the information of whether edges are touched or not
    #False represent not visited yet and we convert the value False to True when we visit 
    #the node 
    #Also storing visited nodes in mst_tour list in ordered way
    
    global mst_touched
    mst_touched = {}
    global mst_tour
    mst_tour = [initial_node]
    for i in nodes:
        mst_touched[i] = False
    mst_touched[initial_node] = True
    
    #Creating empty MST_ARC list to store found edges 
    
    global MST_ARC
    MST_ARC = []

    global adj_list_min_key 
    
    #End of the initialization part
    
    #After this point, computing the following functions until we visited every node
    
    while len(mst_tour) < len(nodes):
    
           
        adj_list_min_key = adjList[min_key]
        
        min_key = check_min(min_key)

        replace_min_key()
        
    
    #Drawing the graph of MST_ARC, which has undirected edges of MST
    
    H = nx.Graph() 
    H.add_edges_from(MST_ARC)
    simple_print_graph(H)

    #We remove the duplicated edges, for example 3,7 and 7,3 is same because 
    #in here we have undirected edges. We only need one of them
    
    MST_ARC = H.edges()
    
    print("\n")
    print(f"MST TOUR is : {mst_tour}\n")
    print(f"Starting and arriving nodes of the each edge in MST : {MST_ARC}\n")
    print("End of prim algorithm\n")
    
    return

#%% Getting weighted edges from MST
#############################

# Function that gets the weights of edges in the MST_ARC from the Weighted_ARC

def get_weighted_arc(A):
    
    global MST_Weighted_ARC
    MST_Weighted_ARC = []
    
    for i in A:
        for j in Weighted_ARC:
            if i[0] == j[0] and i[1] == j[1] or i[1] == j[0] and i[0] == j[1]:
                MST_Weighted_ARC.append(j)
    
    print("\r\n")    
    print(f"MST Arc list with node's weight : {MST_Weighted_ARC}\n")
    
    return MST_Weighted_ARC


#%% Odd Degree 
############################


#Function that gets the MST_ARC as a iput and gives the output of nodes with odd degree
#Odd degree means the node has odd number of neighbor node

def odd_degree(A):

    global odd_degree_nodes
    odd_degree_nodes = []
    nodes_touched = []
    global nodes_degree
    nodes_degree = []

    for i in A:
        nodes_touched.append(i[0])
        nodes_touched.append(i[1])
        
    print("\r\n")    
    print(f"The touched nodes : {nodes_touched}\n")

    
    for i in range(input_nodes):
        nodes_degree.append([i+1, 0])
        nodes_degree[i][1] = nodes_touched.count(i+1)
    
        if nodes_degree[i][1] % 2 != 0:
            odd_degree_nodes.append(nodes_degree[i][0])

    print(f"The node and degree of node in tuple form : {nodes_degree}\n")
    
    print(f"Odd degree nodes : {odd_degree_nodes}\n")

    return(odd_degree_nodes)

#%% Minimum Weight Matching
######################################


#Form Subgraph considering odd degree vertices    

def subgraph():

    global SubARC
    SubARC = []
    global Weighted_SubARC
    Weighted_SubARC = []   
         
    #Adding arcs, which includes odd degree nodes, to the subgraph
    
    for i in range(len(ARC)):            
        x = ARC[i][0]
        y = ARC[i][1]
        if int(x) in odd_degree_nodes and int(y) in odd_degree_nodes:
            SubARC.append((x,y))
           
    #Giving weight to the arcs of the subgraph
            
    for i in Weighted_ARC:
        if (i[0],i[1]) in SubARC:
            Weighted_SubARC.append(i)
                
    H = nx.Graph() 
    H.add_edges_from(SubARC)
    simple_print_graph(H)  
      
    return


#Construct minimum weight matching M

def min_weighted_matching():


    key_odd_degree = {}
    global minimum_matching
    minimum_matching = []
    global minimum_matching_sum_weight
    minimum_matching_sum_weight = 0
                    
    #Sorting the Weighted_SubARC, so we can find the minimum weighted pairs 

    Weighted_SubARC.sort(key = operator.itemgetter(2))   
    
    #Initialization of dictionary for odd degree nodes
    
    for i in odd_degree_nodes:
        key_odd_degree[i] = False
    
    #Picking minimum weight matching in the unvisited odd degree nodes
    #When we visit the odd degree node, value turns into True
    
    for i in Weighted_SubARC:
        if key_odd_degree[i[0]] != True and key_odd_degree[i[1]]  != True:
            minimum_matching.append((i[0],i[1]))
            minimum_matching_sum_weight = minimum_matching_sum_weight + i[2]
            key_odd_degree[i[0]] = True
            key_odd_degree[i[1]] = True
            
    print(f"Minimum weighted matching of odd degree nodes : {minimum_matching}\n")
    print(f"Sum of Weights of Minimum Weight Match : {minimum_matching_sum_weight}\n")   

    return


#%% Unite Minimum Spanning Tree and Minimum Weight Matching
########################################


#Union of Minimum Weight Matching (MWM) and Minimum Spanning Tree (MST)
#Function that unites the MST and MWM edges

def unite(A):

    global TUM_ARC    
    TUM_ARC = []
    
    #Adding arcs from MST T
    for i in A:
        TUM_ARC.append(i)
     
    #Adding arcs from Minimum Weighted Matching M
    for i in minimum_matching:
        TUM_ARC.append(i)
    
    print("\r\n")
    print(f"Combined arcs of MST and MWM : {TUM_ARC}\n")
    
    L = nx.Graph() 
    L.add_edges_from(TUM_ARC)
    simple_print_graph(L)

    return


#%% Euler Tour
#######################
 

#Function for getting list of neighbor nodes for one selected node input, A
def adj_node(A):
    
    next_node_list =[]
    for i in TUM_ARC:
        if A == i[0]:
            next_node_list.append(i[1])
        elif A == i[1]:
            next_node_list.append(i[0])
            
    return next_node_list

#Function for getting set of neighbor nodes for all nodes
def adj_all_node():
    
    global next_node_set
    next_node_set = {}
    
    for i in range(1, input_nodes+1):  
        adj_node(i)
        next_node_set[i] = adj_node(i)
    print(f"Original set of neighbor for all nodes : {next_node_set} \n")
    
    return next_node_set

#Function that sorts the TUM_ARC list
def sort_TUM_ARC(G):
    
    sort_TUM_ARC = []
    for i in G:
        for j in G[i]:
            sort_TUM_ARC.append((i,j))
            
    return sort_TUM_ARC

#Function that check whether there exists a bridge or not
def check_bridge(A):
    
    start_node = list(A)[0]
    color_code = {}
    #Color code represents the availability of nodes, green for available nodes
    #yellow for current nodes and red for visited or unavailable nodes
    for i in A:
        color_code[i] = 'green'
    color_code[start_node] = 'yellow'
    S = [start_node]
    
    while len(S) != 0:

        update_node = S.pop()
        for v in A[update_node]:
            if color_code[v] == 'green':
                color_code[v] = 'yellow'
                S.append(v)
            color_code[update_node] = 'red'
            
    return list(color_code.values()).count('red') == len(A)

#Function that create the Euler Tour mainly
def Euler():
    
    global euler_tour
    
    #Initialization of the necessary list and dictionary by calling corresponding functions
    adj_all_node()
    sort_TUM_ARC(next_node_set)
    
    #We are following Fleury's conditions to decide whether there is Eulerian Graph or not
    if len(odd_degree_nodes) > 2 or len(odd_degree_nodes) == 1:
        return 'Not Eulerian Graph'
    else:
        next_node_set_temp = copy(next_node_set)
        euler_tour = []
        #In case of having exactly 2 odd degree nodes, we start with one of them and finish 
        #with other 
        if len(odd_degree_nodes) == 2:
             update_node = odd_degree_nodes[0]
        
        #In case of having 0 odd degree nodes, we just select the first node
        else:
            update_node = list(next_node_set_temp)[0]
        while len(sort_TUM_ARC(next_node_set)) > 0:
            
            #We append current node and now we are ready to select new value for update node
            #First, we will remove the current node and update note from dictionary and check 
            #if there exists any bridge
            #In case of having bridge we should select that node as update 
            #node(which is also arriving node of next one)
            #If there is no bridge, we will append the removed nodes and select the update node
            #according to elements of dictionary
            
            current_node = update_node
            for update_node in next_node_set_temp[current_node]:
                next_node_set_temp[current_node].remove(update_node)
                next_node_set_temp[update_node].remove(current_node)
                bridge = not check_bridge(next_node_set_temp)
                if bridge:
                    next_node_set_temp[current_node].append(update_node)
                    next_node_set_temp[update_node].append(current_node)
                else:
                     break
            if bridge:
                next_node_set_temp[current_node].remove(update_node)
                next_node_set_temp[update_node].remove(current_node)
                next_node_set_temp.pop(current_node)
            euler_tour.append((current_node, update_node))
    
    print(f"Euler Tour : {euler_tour} \n") 

    H = nx.Graph() 
    H.add_edges_from(euler_tour)
    simple_print_graph(H) 
      
    return euler_tour



#%% Short Cut for Cleaning Euler Tour
################################


#Introduce shortcuts

def short_cut():
    
    global short_cut_sum_weight
    short_cut_sum_weight = 0
    short_cut_weighted_arc = []
    
    #In order to perform short cut operation, we are collecting every starting node into a list
    starting_nodes = []
    
    for i in euler_tour:
        if i[0] not in starting_nodes:
            starting_nodes.append(i[0])

    
    #After collecting starting nodes, we are just removing the repetad ones in order to perform 
    #short cut, which suppose to remove the cycle and repetition between same two nodes
    #We shouldn't return to same node in short cut performed Euler Tour 
    euler_tour_short_cut = []
    
    for i in range(0, len(starting_nodes) - 1):
        euler_tour_short_cut.append((starting_nodes[i], starting_nodes[i+1]))
    euler_tour_short_cut.append((starting_nodes[-1],starting_nodes[0]))

    
    #Summing the all weights of edges of the Euler Tour, which is found after short cut operation
    for i in Weighted_ARC:
        for j in euler_tour_short_cut:
            if j[0] > j[1]:
                if i[0] == j[1] and i[1] == j[0]:
                    short_cut_weighted_arc.append(i)
                    short_cut_sum_weight = short_cut_sum_weight + i[2]
            if j[1] > j[0]:
                if i[0] == j[0] and i[1] == j[1]:
                    short_cut_weighted_arc.append(i)
                    short_cut_sum_weight = short_cut_sum_weight + i[2]
    
    print(f"The short cut Euler Arc : {euler_tour_short_cut} \n")
    print(f"Total Weight of Euler Tour : {short_cut_sum_weight} \n")
           
    H = nx.Graph() 
    H.add_edges_from(euler_tour_short_cut)
    simple_print_graph(H) 

    return

#%% Optimization with Gurobi


# Function that constructs the Minimum Spanning Tree using Formulation
    
def solve_MIP_MST():
    
    global MST_ARC
    MST_ARC = []
    
    #Definition of the model 
    m = gp.Model("mip_MST")

    
    #Variables are associated to edges: the decision will be, for each edge, 
    #if it belongs or not to the Minimum Spanning tree.
    #For this reason, variable are initialized as boolean ones.
    
    X = []
    i = 0

    #Initialization of a matrix for collecting the endpoints of the edges 
    #corresponding to variables
    mat_MST = np.zeros((len(Weighted_ARC_Sorted),2))
    
    while i < len(Weighted_ARC_Sorted):

        u = int(Weighted_ARC_Sorted[i][0])
        v = int(Weighted_ARC_Sorted[i][1])
      
        x = m.addVar(vtype=GRB.BINARY, name=f"x_{u}_{v}")
        
        print(x)
        X.append(x)

        mat_MST[i][0] = u
        mat_MST[i][1] = v
        i = i+1
    print(X)
    print(mat_MST)
        
    
    #Definition of the vector of the costs (weights)
    C = []
    
    i = 0
    while i < len(Weighted_ARC_Sorted):
        C.append(Weighted_ARC_Sorted[i][2])
        i = i+1
        
    print(C)
       


    #Set objective
    m.setObjective(np.dot(C,X), GRB.MINIMIZE)

    #Add constraint 1: 
    #sum x_ij = n-1: the Minimum Spanning Tree contains exactly n - 1 arcs
    
    S = 0
    i = 0
    
    MST_ARC = []
      
    while i < len(X):
         S = S + X[i]
         i = i + 1
      
    m.addConstr(S == input_nodes - 1, "c0")


     
    #Add constraint 2: 
    #no cycles: given any subset of nodes, the number of edges connecting 
    #nodes in the subset has to be smaller than or equal to the cardinality
    #of the subset minus 1.
    
    z=0
    T=0
    combination = 0
    for j in range(2, input_nodes ):
        
        #We generate all the possible combinations of the nodes of the graph.
        #We initialize a counter in order to follow the number of combinations 
        #that have already been computed
        new_list = itertools.combinations(nodes, j)
        print("New arc for the constrain, let's sum it \n\n")
        print("***********************************\n\n")
        print("n combination", combination )
        combination = 0
        for each in new_list:
            combination += 1 
            z = 0
            T = 0
              
            while z < len(Weighted_ARC_Sorted):
                if mat_MST[z][0] in each and mat_MST[z][1] in each:
                     T = T + X[z]

                z = z + 1
                
            m.addConstr( T <= len(each) - 1, "c_MST" + str(z+1)) 
          
          
    
    #Optimization     
    m.optimize()
    
    #We convert the boolean variables whose value is 1 into edges of the MST
    counter = 0
    for v in m.getVars():

         print("Variable name:", v.varName, "   Value:   ", v.x)
           
         if v.x == True:
               
            MST_ARC.append((mat_MST[counter][0], mat_MST[counter][1]))
         counter += 1
               
    print(MST_ARC)              
    print(f'Obj: {m.objVal}')

    print("End of Formulation for MST\n")
    
    #Print of the MST obtained with Formulation
    H = nx.Graph() 
    H.add_edges_from(MST_ARC)
    simple_print_graph(H)     

    return



# Function that constructs the Minimum Weighted Match using Formulation

def solve_MIP_MWM():
    
    global minimum_matching
    minimum_matching = []

    #Definition of the model
    m = gp.Model("mip_MWM")


    #Variables are associated to edges: the decision will be, for each edge, 
    #if it belongs or not to the Minimum Weighted Matching.
    #For this reason, variable are initialized as boolean ones.
    X = []
    i=0
    
    #Initialization of a matrix for collecting the endpoints of the edges 
    #corresponding to variables
    mat_MWM = np.zeros((len(Weighted_SubARC),2))
    
    while i < len(Weighted_SubARC):

        u = int(Weighted_SubARC[i][0])
        v = int(Weighted_SubARC[i][1])
      
        x = m.addVar(vtype=GRB.BINARY, name=f"x_{u}_{v}")
        
        print(x)
        X.append(x)

        mat_MWM[i][0]=u
        mat_MWM[i][1]=v
        
        i=i+1
        
    print(X)
    print(mat_MWM)


    
    #Definition of the vector of the costs (weights)
    W=[]
    
    i=0
    while i < len(Weighted_SubARC):
        W.append(Weighted_SubARC[i][2])
        i = i + 1
    
    print(W)
    

    #Set objective
    m.setObjective(np.dot(W,X), GRB.MINIMIZE)

    #Add constraint: 
    #the sum of all edges ingoing / outgoing from each node has 
    #to be either 0 or 1
      
    S = 0
    z = 0
       
    for i in odd_degree_nodes:
        S = 0
        z = 0     
        
        while z < len(Weighted_SubARC):
       
            if mat_MWM[z][0] == i or mat_MWM[z][1] == i:
              S = S + X[z] 
                
            z = z + 1    
            
        m.addConstr( S  <= 1, "c" + str(z+1))        
    
    
      
    #Add constraint: 
    #The number of arcs outgoing from each odd subset of node has to be at 
    #least 1  
        
    j = 0
    T = 0
    combination = 0
    for r in range(1, len(odd_degree_nodes) + 1):
        if r % 2 != 0:
            
            #We generate all the possible combinations of the nodes of the graph
            new_list = itertools.combinations(odd_degree_nodes, r)
                
            for each in new_list:
                combination += 1 
                j = 0
                T = 0
              
                while j < len(Weighted_SubARC):
                    if mat_MWM[j][0] in each or mat_MWM[j][1] in each:
                            T = T + X[j]
                   
                    j = j + 1
                
                
                m.addConstr( T >= 1, "c_MWM" + str(j+1)) 
                   
      
      
    #Optimization
    m.optimize()

    #We convert the boolean variables whose value is 1 into edges of the MWM
    counter = 0
    for v in m.getVars():
         print("Variable name:", v.varName, "   Value:   ", v.x)
           
         if v.x == True:
               
             minimum_matching.append((mat_MWM[counter][0], mat_MWM[counter][1]))
         counter += 1
               
    print(minimum_matching)              
    print(f'Obj: {m.objVal}')

    print("End of Formulation for MWM\n")
    
    #Print of the MST obtained with Formulation
    H = nx.Graph() 
    H.add_edges_from(minimum_matching)
    simple_print_graph(H)   
      
      
    return
 
