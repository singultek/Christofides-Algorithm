import Christofides_Algorithm
import time
import networkx as nx

#%% User Input Program with Kruskal for the MST and the MWM

def userTSP_Kruskal_MST_Algorithm_MWM():

    start = time.time()

    Christofides_Algorithm.user_input()
    G = nx.Graph() 
    G.add_edges_from(Christofides_Algorithm.ARC)
    Christofides_Algorithm.simple_print_graph(G)
    Christofides_Algorithm.convert(Christofides_Algorithm.A)
    
    Christofides_Algorithm.Kruskal()
    
    Christofides_Algorithm.get_weighted_arc(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.subgraph()
    
    Christofides_Algorithm.min_weighted_matching()
    
    Christofides_Algorithm.unite(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.TUM_ARC)
    Christofides_Algorithm.Euler()
    Christofides_Algorithm.short_cut()

    end = time.time()
    print(f"Time required for the computation is {end - start} seconds \n")

#%% User Input Program with Kruskal for the MST and Formulation for the MWM

def userTSP_Kruskal_MST_Formulation_MWM():
    
    start = time.time()

    Christofides_Algorithm.user_input()
    G = nx.Graph() 
    G.add_edges_from(Christofides_Algorithm.ARC)
    Christofides_Algorithm.simple_print_graph(G)
    Christofides_Algorithm.convert(Christofides_Algorithm.A)
    
    Christofides_Algorithm.Kruskal()
    
    Christofides_Algorithm.get_weighted_arc(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.subgraph()
    
    Christofides_Algorithm.solve_MIP_MWM()
    
    Christofides_Algorithm.unite(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.TUM_ARC)
    Christofides_Algorithm.Euler()
    Christofides_Algorithm.short_cut()

    end = time.time()
    print(f"Time required for the computation is {end - start} seconds \n")

    
#%% User Input Program with Prim for the MST and the MWM

def userTSP_Prim_MST_Algorithm_MWM():

    start = time.time()

    Christofides_Algorithm.user_input()
    G = nx.Graph() 
    G.add_edges_from(Christofides_Algorithm.ARC)
    Christofides_Algorithm.simple_print_graph(G)
    Christofides_Algorithm.convert(Christofides_Algorithm.A)
    
    Christofides_Algorithm.Prim()
    
    Christofides_Algorithm.get_weighted_arc(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.subgraph()
    
    Christofides_Algorithm.min_weighted_matching()
    
    Christofides_Algorithm.unite(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.TUM_ARC)
    Christofides_Algorithm.Euler()
    Christofides_Algorithm.short_cut()

    end = time.time()
    print(f"Time required for the computation is {end - start} seconds \n")

#%% User Input Program with Prim for the MST and Formulation for the MWM

def userTSP_Prim_MST_Formulation_MWM():

    start = time.time()

    Christofides_Algorithm.user_input()
    G = nx.Graph() 
    G.add_edges_from(Christofides_Algorithm.ARC)
    Christofides_Algorithm.simple_print_graph(G)
    Christofides_Algorithm.convert(Christofides_Algorithm.A)
    
    Christofides_Algorithm.Prim()
    
    Christofides_Algorithm.get_weighted_arc(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.subgraph()
    
    Christofides_Algorithm.solve_MIP_MWM()
    
    Christofides_Algorithm.unite(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.TUM_ARC)
    Christofides_Algorithm.Euler()
    Christofides_Algorithm.short_cut()

    end = time.time()
    print(f"Time required for the computation is {end - start} seconds \n")    
 

#%% User Input Program with Formulation for the MST and Algorithm for the MWM

def userTSP_Formulation_MST_Algorithm_MWM():

    start = time.time()

    Christofides_Algorithm.user_input()
    G = nx.Graph() 
    G.add_edges_from(Christofides_Algorithm.ARC)
    Christofides_Algorithm.simple_print_graph(G)
    Christofides_Algorithm.convert(Christofides_Algorithm.A)
    
    Christofides_Algorithm.solve_MIP_MST()
    
    Christofides_Algorithm.get_weighted_arc(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.subgraph()
    
    Christofides_Algorithm.min_weighted_matching()
    
    Christofides_Algorithm.unite(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.TUM_ARC)
    Christofides_Algorithm.Euler()
    Christofides_Algorithm.short_cut()

    end = time.time()
    print(f"Time required for the computation is {end - start} seconds \n")


#%% User Input Program with Formulation for the MST and the MWM

def userTSP_Formulation_MST_Formulation_MWM():
    
    start = time.time()

    Christofides_Algorithm.user_input()
    G = nx.Graph() 
    G.add_edges_from(Christofides_Algorithm.ARC)
    Christofides_Algorithm.simple_print_graph(G)
    Christofides_Algorithm.convert(Christofides_Algorithm.A)
    
    Christofides_Algorithm.solve_MIP_MST()
    
    Christofides_Algorithm.get_weighted_arc(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.subgraph()
    
    Christofides_Algorithm.solve_MIP_MWM()
    
    Christofides_Algorithm.unite(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.TUM_ARC)
    Christofides_Algorithm.Euler()
    Christofides_Algorithm.short_cut()

    end = time.time()
    print(f"Time required for the computation is {end - start} seconds \n")    


#%% TSP Input Program with Kruskal for the MST and the MWM

def TSPlib_Kruskal_MST_Algorithm_MWM():

    start = time.time()

    Christofides_Algorithm.tsp_input()
    G = nx.Graph() 
    G.add_edges_from(Christofides_Algorithm.ARC)
    Christofides_Algorithm.simple_print_graph(G)
    Christofides_Algorithm.convert(Christofides_Algorithm.A)
    
    Christofides_Algorithm.Kruskal()
    
    Christofides_Algorithm.get_weighted_arc(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.subgraph()
    
    Christofides_Algorithm.min_weighted_matching()
    
    Christofides_Algorithm.unite(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.TUM_ARC)
    Christofides_Algorithm.Euler()
    Christofides_Algorithm.short_cut()

    end = time.time()
    print(f"Time required for the computation is {end - start} seconds \n")

#%% TSP Input Program with Kruskal for the MST and Formulation for the MWM

def TSPlib_Kruskal_MST_Formulation_MWM():
    
    start = time.time()

    Christofides_Algorithm.tsp_input()
    G = nx.Graph() 
    G.add_edges_from(Christofides_Algorithm.ARC)
    Christofides_Algorithm.simple_print_graph(G)
    Christofides_Algorithm.convert(Christofides_Algorithm.A)
    
    Christofides_Algorithm.Kruskal()
    
    Christofides_Algorithm.get_weighted_arc(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.subgraph()
    
    Christofides_Algorithm.solve_MIP_MWM()
    
    Christofides_Algorithm.unite(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.TUM_ARC)
    Christofides_Algorithm.Euler()
    Christofides_Algorithm.short_cut()

    end = time.time()
    print(f"Time required for the computation is {end - start} seconds \n")


#%% TSP Input Program with Prim for the MST and the MWM

def TSPlib_Prim_MST_Algorithm_MWM():

    start = time.time()

    Christofides_Algorithm.tsp_input()
    G = nx.Graph() 
    G.add_edges_from(Christofides_Algorithm.ARC)
    Christofides_Algorithm.simple_print_graph(G)
    Christofides_Algorithm.convert(Christofides_Algorithm.A)
    
    Christofides_Algorithm.Prim()
    
    Christofides_Algorithm.get_weighted_arc(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.subgraph()
    
    Christofides_Algorithm.min_weighted_matching()
    
    Christofides_Algorithm.unite(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.TUM_ARC)
    Christofides_Algorithm.Euler()
    Christofides_Algorithm.short_cut()

    end = time.time()
    print(f"Time required for the computation is {end - start} seconds \n")
    

#%% TSP Input Program with Prim for the MST and Formulation for the MWM

def TSPlib_Prim_MST_Formulation_MWM():
    
    start = time.time()

    Christofides_Algorithm.tsp_input()
    G = nx.Graph() 
    G.add_edges_from(Christofides_Algorithm.ARC)
    Christofides_Algorithm.simple_print_graph(G)
    Christofides_Algorithm.convert(Christofides_Algorithm.A)
    
    Christofides_Algorithm.Prim()
    
    Christofides_Algorithm.get_weighted_arc(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.subgraph()
    
    Christofides_Algorithm.solve_MIP_MWM()
    
    Christofides_Algorithm.unite(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.TUM_ARC)
    Christofides_Algorithm.Euler()
    Christofides_Algorithm.short_cut()

    end = time.time()
    print(f"Time required for the computation is {end - start} seconds \n")
    

#%% TSP Input Program with Formulation for the MST and Algorithm for the MWM

def TSPlib_Formulation_MST_Algorithm_MWM():
    
    start = time.time()

    Christofides_Algorithm.tsp_input()
    G = nx.Graph() 
    G.add_edges_from(Christofides_Algorithm.ARC)
    Christofides_Algorithm.simple_print_graph(G)
    Christofides_Algorithm.convert(Christofides_Algorithm.A)
    
    Christofides_Algorithm.solve_MIP_MST()
    
    Christofides_Algorithm.get_weighted_arc(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.subgraph()
    
    Christofides_Algorithm.min_weighted_matching()
    
    Christofides_Algorithm.unite(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.TUM_ARC)
    Christofides_Algorithm.Euler()
    Christofides_Algorithm.short_cut()

    end = time.time()
    print(f"Time required for the computation is {end - start} seconds \n")
    
#%% TSP Input Program with Formulation for the MST and the MWM

def TSPlib_Formulation_MST_Formulation_MWM():
    
    start = time.time()

    Christofides_Algorithm.tsp_input()
    G = nx.Graph() 
    G.add_edges_from(Christofides_Algorithm.ARC)
    Christofides_Algorithm.simple_print_graph(G)
    Christofides_Algorithm.convert(Christofides_Algorithm.A)
    
    Christofides_Algorithm.solve_MIP_MST()
    
    Christofides_Algorithm.get_weighted_arc(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.subgraph()
    
    Christofides_Algorithm.solve_MIP_MWM()
    
    Christofides_Algorithm.unite(Christofides_Algorithm.MST_ARC)
    Christofides_Algorithm.odd_degree(Christofides_Algorithm.TUM_ARC)
    Christofides_Algorithm.Euler()
    Christofides_Algorithm.short_cut()

    end = time.time()
    print(f"Time required for the computation is {end - start} seconds \n")

   