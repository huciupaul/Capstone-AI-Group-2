# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from collections import deque
import modularity as modularity
import _divide as _divide

def partition(network, extreme_nodes, refine=True):
    '''
    Cluster a network into several modules
    using modularity maximization by spectral methods.

    Supports directed and undirected networks, with weighted or unweighted edges.

    See:

    Newman, M. E. J. (2006). Modularity and community structure in networks.
    Proceedings of the National Academy of Sciences of the United States of America,
    103(23), 8577–82. https://doi.org/10.1073/pnas.0601602103

    Leicht, E. A., & Newman, M. E. J. (2008). Community Structure in Directed Networks.
    Physical Review Letters, 100(11), 118703. https://doi.org/10.1103/PhysRevLett.100.118703

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network of interest
    refine: Boolean
        Whether refine the `s` vector from the initial clustering
        by repeatedly moving nodes to maximize modularity

    Returns
    -------
    dict
        A dictionary that saves membership.
        Key: node label; Value: community index
    '''
    ## preprocessing
    
    
    # Create a mapping from original node names to integer labels
    name_to_label = {name: i for i, name in enumerate(network.nodes(data="node_name", default=None), start=1)}

    # Convert the node labels to integers while preserving the original names
    nx.relabel_nodes(network, name_to_label, copy=False)
    
    B = modularity.get_base_modularity_matrix(network)
    # Initializes a queue with initial community indices [0, 1]. These represent the first two communities to be processed
    divisible_community = deque([0, 1])
    
    # Initialize the community dictionary
    community_dict = {name: 0 for name in network.nodes()}  # Assign community index 0 to all nodes
    
    
    #print("community dict before adding extreme nodes: ", community_dict)
    # Assign a different community index to extreme nodes
    for name in extreme_nodes:
        community_dict[name] = 1
        
    #print("community dict: ", community_dict)     


    comm_counter = 1

    while len(divisible_community) > 0:
        ## get the first divisible comm index out
        comm_index = divisible_community.popleft()
        
        #print("Processing community:", comm_index)
         #Attempts to split the nodes of comm_index into two groups (g1_nodes and the rest).
        g1_nodes, comm_nodes = _divide._divide(network, community_dict, comm_index, B, refine)
        if g1_nodes is None:
            ## indivisible, go to next
            #print("Community", comm_index, "is indivisible.")
            continue
        ## Else divisible, obtain the other group g2
        #print("Community", comm_index, "is divisible.")
        
        #### Get the subgraphs (sub-communities)
        #print("g1_nodes: ", g1_nodes)
        #Creates two subgraphs, g1 and g2, representing the two new communities.
        g1 = network.subgraph(g1_nodes)
        g2 = network.subgraph(set(comm_nodes).difference(set(g1_nodes)))
        parent = "%d"%comm_index

        ## add g1, g2 to tree and divisible list
        #Assigns a new community index (comm_counter) to g1 nodes and adds it to the queue.
        comm_counter += 1
        divisible_community.append(comm_counter)
        ## update community
        for u in g1:
            community_dict[u] = comm_counter

        comm_counter += 1
        divisible_community.append(comm_counter)
        ## update community
        for u in g2:
            community_dict[u] = comm_counter

    # corrects partition numbering to be in 0,...,M-1 (matches python implementation
    #   of the Louvain algorithm), and restore names of nodes
    # old to new numbering
    #Creates a mapping from old community indices to consecutive integers starting from 0.
    old_to_new = {}
    new_val = 0
    for v in set(community_dict.values()):
        old_to_new[v] = new_val
        new_val += 1
    
    
    # Remap community indices using the old_to_new dictionary
    optimal_partition = {}
    for k in community_dict:
        optimal_partition[k] = old_to_new[community_dict[k]]
    #Identifies the unique communities associated with the provided extreme_nodes.
    extreme_communities = []
    for key, value in optimal_partition.items():
        if key in extreme_nodes:
            extreme_communities.append(value)

    #Removes duplicates from the list of extreme communities.  
    extreme_communities = np.unique(np.array(extreme_communities))
    
    #print("These are the indentified extreme communities: ", extreme_communities)

    # Create a dictionary to map the original node names to community indices
    #optimal_partition = {name: community_dict[name] for name in network.nodes()}
    #Outputs the final node-to-community mapping and the identified extreme communities.
    return optimal_partition, extreme_communities