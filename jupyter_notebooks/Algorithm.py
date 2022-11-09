#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 23:21:08 2022

@author: puneeth

Developing Algortihm 
"""

import numpy as np 

#This fuctions gets a path from the start node to the end node such all the intermediate nodes are before the younger of the two nodes.
def shortest_path(neighbour_nodes, start, end): 
    younger = min(start, end)
    path_sets = [ [start] ]  #This will keep track of all possible paths
    while True: #Keep running this loop until you return a value. As long as there is a MRCA, this won't become an infinite loop. The possible path = start -> MRCA -> end.
        path_sets_temp = [] #temporary path_sets list to keep track of new things while going through the previous path_sets. 
        for path in path_sets: 
            last_node = path[-1] #Get the last nodes to figure out what the next nodes can be. 
            nxt_nodes = list([x for x in neighbour_nodes[last_node] if x > younger]) #Creating a new path for all neigbhour nodes of the last node that are atleast older than the younger node of the start and end nodes.
            #For each possible next node, we create a new path to keep track of it. 
            for nxt_node in nxt_nodes: 
                new_path = path + [ nxt_node ] #Creating the new path.
                path_sets_temp += [ new_path ] #Adding it to the set of paths. 
                if nxt_node == end : 
                    return new_path #Checking if any of paths reached the end node. In which case, we got the path!
                else : 
                    path_sets = path_sets_temp 
    

 #Take input as as set of times of the nodes [ t0, t1, t2, t3,...,t8 ] and a edge matrix [ [1,0,1,0,1],[] ]. The input times have to be ordered accordint to increasing values. 
[ t0, t1, t2, t3, t4, t5, t6, t7, t8 ]  = [ 0, 0, 0, 20, 30, 70, 100, 140, 190 ]
node_times = { 0:t0, 1:t1, 2:t2, 3:t3, 4:t4, 5:t5, 6:t6, 7:t7, 8:t8 }
edge_mat = [ [0,0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[1,1,0,0,0,1,0,0,0],[0,0,1,0,0,0,1,1,0],[0,0,0,1,0,0,1,0,1],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,1,0,1],[0,0,0,0,0,0,0,1,0] ]

N_nodes = len(node_times) #Number of nodes 

#Create a set of neighbour nodes
neighbour_nodes = { x:[ y for y in range(N_nodes) if edge_mat[x][y] == 1 ] for x in range(N_nodes) } 


#Create a list of parent nodes, nodes with singles parents and nodes with two parents.
parent_nodes_weighted = {} #For each nodes gives a list containing all the parent nodes. 
one_parent_nodes = [] #List of all nodes that have a single parent
two_parent_nodes = [] #List of all nodes that have two parents

for i in range(N_nodes): 
    parent_nodes_temp = {}
    for j in range(i+1 ,N_nodes): 
        #Decide if j is a parent of i. If so add it to the parent list
        if edge_mat[i][j] == 1: 
            parent_nodes_temp[ j ] = node_times[j] - node_times[i] #The weight is the variance which here is equal to the difference between the times. 
        parent_nodes_weighted[i] = parent_nodes_temp
        
    #Decide if i is a one_parent_node or a two_parent_node
    if len(list(parent_nodes_temp.keys())) == 1 : 
        one_parent_nodes += [i]
    elif len(list(parent_nodes_temp.keys())) == 2 : 
        two_parent_nodes += [i]


# Make a list of all intermediate nodes for two parent nodes 
intermediate_paths = { x: shortest_path(neighbour_nodes, min(list(parent_nodes_weighted[x].keys()) ), max( list(parent_nodes_weighted[x].keys()) )) for x in two_parent_nodes }




edge_set_weights = { (x,y):parent_nodes_weighted[x][y] for x in parent_nodes_weighted.keys() for y in parent_nodes_weighted[x].keys() }#Get the set of edges and the time-length
edge_set = list(edge_set_weights.keys()) #Convert the keys dictionary into a list so we can use the sort function.
edge_set.sort(reverse=True) #Arange them in decreasing order (the oldest one first) of the child node, breaking degeneracy with decreasing order of parent node 

Cov_Mat = { x:{ y: np.nan for y in edge_set } for x in edge_set } #Initialize the Variance - Covariance Matrix

#Go through every pair of edges and compute their covariance. 
for edge1_ind in range(len(edge_set)): 
    for edge2_ind in range(edge1_ind + 1): 
        edge1 = edge_set[edge1_ind]
        edge2 = edge_set[edge2_ind]
        
        # When both are single parent edges, then they are independent and normally distributed with variance equal to their length
        if edge1[0] in one_parent_nodes and edge2[0] in one_parent_nodes:
            if edge1 == edge2: 
                Cov_Mat[edge1][edge2] = node_times[edge1[1]] - node_times[edge1[0]]
            else:
                Cov_Mat[edge1][edge2] = 0
        
        #When edge1 is a two parent node 
        elif edge1[0] in two_parent_nodes: 
            pnodes = list(parent_nodes_weighted[edge1[0]].keys()) #The two parent nodes of edge1[0] (which is the offspring node of edge1)
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
                Cov_Mat[ edge1 ][ edge2 ] = -edge_set_weights[ edge1 ]*Cov_Int/( edge_set_weights[ (offspring_node, younger_pnode) ] + edge_set_weights[ (offspring_node, older_pnode) ] ) #Eq 13 and 16
            
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
                #If the edges are equal, we use Eq 12 or Eq 8
                if edge1 == edge2 : 
                    Cov_Mat[ edge1 ][ edge2 ] = edge_set_weights[younger_pedge]*edge_set_weights[older_pedge]/( edge_set_weights[younger_pedge] + edge_set_weights[older_pedge] )  +  ( edge_set_weights[edge1]  / ( edge_set_weights[younger_pedge] + edge_set_weights[older_pedge] )  )**2*Var_Int 
                #If the edges are the two parent edges of a loop, we use Eq 14
                elif edge1[1] != edge2[1] : 
                    Cov_Mat[ edge1 ][ edge2 ] = edge_set_weights[younger_pedge]*edge_set_weights[older_pedge]/( edge_set_weights[younger_pedge] + edge_set_weights[older_pedge] )  -   edge_set_weights[edge1]*edge_set_weights[edge2]*Var_Int  / ( edge_set_weights[younger_pedge] + edge_set_weights[older_pedge] )**2 
                #Code Check
                else: 
                    raise TypeError('Provided edge1 is a two parent edge ')
                
                
        
        #When edge1 is a one-parent node but edge2 is not. Same as the one of the cases above. 
        elif edge1[0] in one_parent_nodes and edge2[0] in two_parent_nodes: 
            pnodes = list(parent_nodes_weighted[edge2[0]].keys()) #The two parent nodes of edge1[0] (which is the offspring node of edge1)
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
            Cov_Mat[ edge1 ][ edge2 ] = -edge_set_weights[ edge2 ]*Cov_Int/( edge_set_weights[ (offspring_node, younger_pnode) ] + edge_set_weights[ (offspring_node, older_pnode) ] )
        
        else:
            raise TypeError(' Edge Combination not accounted for ')
        Cov_Mat[edge2][edge1] = Cov_Mat[edge1][edge2]
            


    
#Now we compute covariances. 