#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 20:25:01 2022

@author: puneeth
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pyslim
import tskit
import numpy as np
import argutils #This is not an installed package but is available as a folder 
import matplotlib.pyplot as plt
import networkx as nx 

#This fuctions gets a path from the start node to the end node such all the intermediate nodes are before the younger of the two nodes.
def shortest_path(DiGrph, start, end): 
    younger = min( [start, end ] ,key = lambda n:ARG.nodes()[n]['time'] ) #In this code this will always be the start by design. 
    path_sets = [ [start] ]  #This will keep track of all possible paths
    while True: #Keep running this loop until you return a value. As long as there is a MRCA, this won't become an infinite loop. The possible path = start -> MRCA -> end.
        path_sets_temp = [] #temporary path_sets list to keep track of new things while going through the previous path_sets. 
        for path in path_sets: 
            last_node = path[-1] #Get the last nodes to figure out what the next nodes can be. 
            neighbour_nodes = list(ARG.successors(last_node)) + list(ARG.predecessors(last_node))
            nxt_nodes = list([ x for x in neighbour_nodes if ARG.nodes()[x]['time'] > ARG.nodes()[younger]['time'] ]) #Creating a new path for all neigbhour nodes of the last node that are atleast older than the younger node of the start and end nodes.
            #For each possible next node, we create a new path to keep track of it. 
            for nxt_node in nxt_nodes: 
                new_path = path + [ nxt_node ] #Creating the new path.
                path_sets_temp += [ new_path ] #Adding it to the set of paths. 
                if nxt_node == end : 
                    return new_path #Checking if any of paths reached the end node. In which case, we got the path!
                else : 
                    path_sets = path_sets_temp 


#This is to calculate the MLE for dispersal and the ancestral location copied as it is from Matt's Code.                     
def mles(sample_locations, Cov_Mat, mean_center=True):    
    '''
    mle MRCA location and dispersal rate from sample locations and shared time matrix under branching Brownian motion
    '''
    
    n,d = np.array(sample_locations).shape #number of samples and spatial dimension
    ones = np.ones(n)
    
    if mean_center:
        M = np.identity(n) - [[1/n for _ in range(n)] for _ in range(n)]; M=M[:-1]#matrix for mean centering and dropping one sample
        locations = np.matmul(M, sample_locations) #mean center locations and drop last sample 
        times = np.matmul(M, np.matmul(Cov_Mat, np.transpose(M))) #mean center times
        ones = ones[:-1]
    
    T = np.linalg.pinv(np.array(times)) #pseudo-inverse of mean centered time matrix
    
    ahat = np.zeros(d)
    if not mean_center:
        # find MLE MRCA location (eqn 5.6 Harmon book)
        a1 = np.matmul(np.matmul(ones, T), ones.reshape(-1,1))
        a2 = np.matmul(np.matmul(ones, T), locations)
        ahat = a2/a1
    
    # find MLE dispersal rate (eqn 5.7 Harmon book)
    x = locations - ahat * ones.reshape(-1,1)
    Sigma = np.matmul(np.matmul(np.transpose(x), T), x) / (n - 1)   
        
    return ahat, Sigma

ARG = nx.DiGraph() #Initialize an empty DirectedGraph which will store the information of the ARG 
[ t0, t1, t2, t3, t4, t5, t6, t7, t8 ]  = [ 0, 0, 0, 20, 30, 70, 100, 140, 190 ]
node_times = { 0:t0, 1:t1, 2:t2, 3:t3, 4:t4, 5:t5, 6:t6, 7:t7, 8:t8 }
edge_mat = [ [0,0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[1,1,0,0,0,1,0,0,0],[0,0,1,0,0,0,1,1,0],[0,0,0,1,0,0,1,0,1],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,1,0,1],[0,0,0,0,0,0,0,1,0] ]

N_nodes = len(node_times)

#Adding all the nodes to ARG with attributes as their time and location. Note that for inference the location information would only be available for the present day nodes. 
for nd in node_times:  
    ARG.add_node(nd, time = node_times[nd])

#Adding the edges along with the time between the parent and offspring of the edge. 
for nd_i in node_times:
    for nd_j in range(nd_i+1 ,N_nodes):
        if ARG.has_node(nd_i) and ARG.has_node(nd_j) :
            if edge_mat[nd_i][nd_j] == 1 : 
                ARG.add_edge( nd_i, nd_j, time_len = ARG.nodes()[nd_j]['time'] - ARG.nodes()[nd_i]['time']  )
        else : 
            raise TypeError("One of the nodes in the edge is not avaialble.")



#Create a list of nodes with singles parents and nodes with two parents.
one_parent_nodes = [] #List of all nodes that have a single parent
two_parent_nodes = [] #List of all nodes that have two parents

for nd in ARG.nodes(): 
    parents = list(ARG.successors(nd))
    if len(parents) == 1 : 
        one_parent_nodes += [nd]
    elif len(parents) == 2 : 
        two_parent_nodes += [nd]
    else:
        print(nd,len(parents))


# Make a list of all intermediate nodes for two parent nodes 
intermediate_paths = { x: shortest_path(ARG, min(list(ARG.successors(x)),key = lambda n:ARG.nodes()[n]['time']  ), max( list(ARG.successors(x)),key = lambda n:ARG.nodes()[n]['time']  ) ) for x in two_parent_nodes }


edge_set = list(ARG.edges()) #Convert the keys dictionary into a list so we can use the sort function.
edge_set.sort(reverse=True, key = lambda n: (n[0],n[1]) ) #Arange them in decreasing order (the oldest one first) of the child node, breaking degeneracy with decreasing order of parent node 

Cov_Mat = { x:{ y: np.nan for y in edge_set } for x in edge_set } #Initialize the Variance - Covariance Matrix

#Go through every pair of edges and compute their covariance. 
for edge1_ind in range(len(edge_set)): 
    for edge2_ind in range(edge1_ind + 1): 
        edge1 = edge_set[edge1_ind]
        edge2 = edge_set[edge2_ind]
        
        # When both are single parent edges, then they are independent and normally distributed with variance equal to their length
        if edge1[0] in one_parent_nodes and edge2[0] in one_parent_nodes:
            if edge1 == edge2: 
                Cov_Mat[edge1][edge2] = ARG.edges()[edge1]['time_len']
            else:
                Cov_Mat[edge1][edge2] = 0
        
        #When edge1 is a two parent node 
        elif edge1[0] in two_parent_nodes: 
            pnodes = list(ARG.successors(edge1[0])) #The two parent nodes of edge1[0] (which is the offspring node of edge1)
            offspring_node = edge1[0] 
            younger_pnode = min(pnodes) #Figure out the younger parent node. The p stands for parent
            older_pnode = max(pnodes) #Figure out the older parent node. 
            younger_pedge = (offspring_node,younger_pnode) #Edge with younger parent and the offspring
            older_pedge = (offspring_node,older_pnode) #Edge with older parent and the offspring 
            
            intermediate_path = intermediate_paths[edge1[0]] #Get the intermediate path between the two parents nodes of edge1[0] (which is the offspring node)
            
            #First consider the case when the offspring nodes of the two edges are different. So the cases where the covariance formula use Eq 13 and Eq 16 
            if edge1[0] != edge2[0] :
                Cov_Int = 0 #To compute the Covariance of edge two with the path between of the younger parent and older parent. 
                #Go through the nodes in the intermediate path between the two parent nodes sequentially. 
                for i in range( len(intermediate_path) -1 ):
                    #To figure out the edges in the path 
                    st = intermediate_path[i] 
                    nxt = intermediate_path[i+1]
                    younger_stnxt = min(st,nxt) #younger of the start and next nodes
                    older_stnxt = max(st,nxt) #older of the start and next nodes
                    #If the start node is the younger one, then we are going in the edge (which goes from the younger to the older node) is along the direction of the path from the younger parent node to the older parent not. 
                    if younger_stnxt == st: 
                        Cov_Int += Cov_Mat[edge2][ (younger_stnxt,older_stnxt) ] 
                    #In the other case, the edge and the intermediate path are in opposite directions. 
                    elif younger_stnxt == nxt: 
                        Cov_Int -= Cov_Mat[edge2][ (younger_stnxt,older_stnxt) ]
                    #Just a coding check 
                    else: 
                        raise TypeError('Error deciding alignment of the edge and the intermediate path')
                    
                #The intermediate path starts at the younger node. If the edge1 has the younger parent node, then the intermediate goes from the edge1 parent node to the other parent node. 
                if edge1[1] == younger_pnode: 
                    Cov_Int = Cov_Int
                #If edge1 has the older parent node, then the intermediate node is in the opposite direction. 
                elif edge1[1] == older_pnode: 
                    Cov_Int = -Cov_Int
                #Code Check
                else: 
                    raise TypeError(' Parent Nodes do not match ')
                
                #Calculating the time lengths of all edges involved. NOTE: edge1 is definitely either the younger pedge or the older pedge. But since we don't know which we need to keep track of it. 
                e1_wgt = ARG.edges()[ edge1 ]['time_len'] #edge 1 weight 
                y_ped_wgt = ARG.edges()[ younger_pedge ]['time_len'] #younger pedge weight
                o_ped_wgt = ARG.edges()[ older_pedge ]['time_len'] #older pedge weight
                
                Cov_Mat[ edge1 ][ edge2 ] = - e1_wgt*Cov_Int/( y_ped_wgt + o_ped_wgt ) #Eq 13 and 16
            
            #If the offspring nodes are equal, so edge1 and edge2 are the two parent edges or the two are the same edge. 
            else: 
                Var_Int = 0 #To compute the variance of the intermediate path between the two parent nodes. 
                for i in range( len(intermediate_path) -1 ):
                    for j in range( len(intermediate_path) -1 ):
                        st1 = intermediate_path[i]
                        nxt1 = intermediate_path[i+1]
                        younger_stnxt1 = min(st1,nxt1)
                        older_stnxt1 = max(st1,nxt1)
                        
                        st2 = intermediate_path[j]
                        nxt2 = intermediate_path[j+1]
                        younger_stnxt2 = min(st2,nxt2)
                        older_stnxt2 = max(st2,nxt2)
                        
                        multiplicative_factor = 1 #Edges in this algorithm are directed from the younger to the older. But an edge in the intermediate path maybe from an older to a younger node. So, we might need a corrective factor of -1 if one edge is in the direction and one is against the direction of the intermediate path. 
                        if younger_stnxt1 == st1:
                            multiplicative_factor *= -1                             
                        if younger_stnxt2 == st2: 
                            multiplicative_factor *= -1                              
                            
                        Var_Int += multiplicative_factor*Cov_Mat[ (younger_stnxt1,older_stnxt1) ][ (younger_stnxt2,older_stnxt2) ]
                
                #Calculating the time lengths of all edges involved. NOTE: edge1 is definitely either the younger pedge or the older pedge. But since we don't know which we need to keep track of it. 
                e1_wgt = ARG.edges()[ edge1 ]['time_len'] #edge 1 weight 
                y_ped_wgt = ARG.edges()[ younger_pedge ]['time_len'] #younger pedge weight
                o_ped_wgt = ARG.edges()[ older_pedge ]['time_len'] #older pedge weight
                
                #If the edges are equal, we use Eq 12 or Eq 8
                if edge1 == edge2 : 
                    Cov_Mat[ edge1 ][ edge2 ] = y_ped_wgt*o_ped_wgt/(y_ped_wgt + o_ped_wgt) + ( e1_wgt/(y_ped_wgt + o_ped_wgt))**2*Var_Int 
                #If the edges are the two parent edges of a loop, we use Eq 14
                elif edge1[1] != edge2[1] : 
                    Cov_Mat[ edge1 ][ edge2 ] = y_ped_wgt*o_ped_wgt/(y_ped_wgt + o_ped_wgt) - y_ped_wgt*o_ped_wgt*Var_Int / (y_ped_wgt + o_ped_wgt)**2 
                #Code Check
                else: 
                    raise TypeError('Provided edge1 is a two parent edge ')
                
                
        
        #When edge1 is a one-parent node but edge2 is not. Same as the one of the cases above. 
        elif edge1[0] in one_parent_nodes and edge2[0] in two_parent_nodes: 
            pnodes = list(ARG.successors(edge2[0])) #The two parent nodes of edge1[0] (which is the offspring node of edge1)
            offspring_node = edge2[0]
            younger_pnode = min(pnodes)
            older_pnode = max(pnodes)
            intermediate_path = intermediate_paths[edge2[0]] #Get the intermediate path between the two parents nodes of edge1[0] (which is the offspring node)
            
            Cov_Int = 0 
            for i in range( len(intermediate_path) -1 ):
                st = intermediate_path[i]
                nxt = intermediate_path[i+1]
                younger_stnxt = min(st,nxt)
                older_stnxt = max(st,nxt)
                if younger_stnxt == st:
                    Cov_Int += Cov_Mat[edge1][ (younger_stnxt,older_stnxt) ]
                elif younger_stnxt == nxt: 
                    Cov_Int -= Cov_Mat[edge1][ (younger_stnxt,older_stnxt) ]
                    
            if edge2[1] == younger_pnode: 
                Cov_Int = Cov_Int                
                        
            elif edge2[1] == older_pnode: 
                Cov_Int = -Cov_Int
                
            else: 
                raise TypeError(' Parent Nodes do not match ')
            
            #Calculating the time lengths of all edges involved. NOTE: edge2 is definitely either the younger pedge or the older pedge. But since we don't know which we need to keep track of it. 
            e2_wgt = ARG.edges()[ edge2 ]['time_len'] #edge 1 weight 
            y_ped_wgt = ARG.edges()[ (offspring_node, younger_pnode) ]['time_len'] #younger pedge weight
            o_ped_wgt = ARG.edges()[ (offspring_node, older_pnode) ]['time_len'] #older pedge weight
            
            Cov_Mat[ edge1 ][ edge2 ] = -e2_wgt*Cov_Int/( y_ped_wgt + o_ped_wgt ) #Eq16
        
        else:
            raise TypeError(' Edge Combination not accounted for ')
        Cov_Mat[edge2][edge1] = Cov_Mat[edge1][edge2]

#Now we compute the Covariance Matrix for the current locations of the sample.
sample = [0,1,2] #The current sample here is 0 , 1 and 2
Cov_Mat_True = { x: {y: np.nan for y in sample} for x in sample } #The Covariance Matrix for the locations of the samples
GMRCA = max( list(ARG.nodes()) , key= lambda n: ARG.nodes()[n]['time'] ) #The oldest node is assumed to be the Grand MRCA 

path_to_GMRCA = { x : shortest_path(ARG, x, GMRCA)  for x in sample} #Get the the path from the samples to the GMRCA. The shortest path is a list of nodes. 
path_to_GMRCA = { x : [ (path_to_GMRCA[x][ind],path_to_GMRCA[x][ind+1]) for ind in range(len(path_to_GMRCA[x])-1) ] for x in sample }#The shortherst path as list of edges. 
for sam1 in sample: 
    for sam2 in sample: 
        path1 = path_to_GMRCA[sam1] 
        path2 = path_to_GMRCA[sam2]
        CV_s1s2 = 0 #Initializing the variable that will compute the covariance between sample 1 and sample 2 locations
        # Using the fact that covariance is colinear in both variables.
        for ed1 in path1:
            for ed2 in path2:
                CV_s1s2 += Cov_Mat[ed1][ed2]
        Cov_Mat_True[sam1][sam2] = CV_s1s2 
        
#Now, we compute the MLE for dispersal rate and ancestral locations 
Cov_Mat_True = [ [ Cov_Mat_True[x][y] for y in sample ] for x in sample ]
locations = [ [2,2], [10,4], [7,5] ]

a_hat, Sig = mles(locations, Cov_Mat_True)


#Finally, we plot the ARG 
pos = {} 
i=1
for nd in ARG.nodes():
    if ARG.nodes()[nd]['time'] ==0 : 
        pos[nd] = (i , ARG.nodes()[nd]['time'] )
        i += 2
    else: 
        x_loc = np.average([ pos[x][0] for x in ARG.predecessors(nd) ]) 
        pos[nd] = ( x_loc , ARG.nodes()[nd]['time'] )
# pos = {0:(10, 10),
#  1:(7.5, 7.5), 2:(12.5, 7.5),
#  3:(6, 6), 4:(9, 6),
#  5:(11, 6), 6:(14, 6), 7:(17, 6)}

nx.draw_networkx(ARG, pos = pos, arrows = True, node_shape = "s", node_color = "white")
plt.title("ARG")
# plt.savefig(“Output/plain organogram using networkx.jpeg”, dpi = 300)
plt.show()
