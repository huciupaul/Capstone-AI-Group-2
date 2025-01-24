# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse import issparse
import modularity as modularity


#The function divides a specific community in the network into two subgroups based on modularity optimization, leveraging spectral methods.
def _divide(network, community_dict, comm_index, B, refine=False):
    '''
    Bisection of a community in `network`.

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network of interest

    Returns
    -------
    tuple
        If the given community is indivisible, return (None, None)
        If the given community is divisible, return a tuple where
        the 1st element is a node list for the 1st sub-group and
        the 2nd element is a node list for the original group
    '''
    #Extract nodes in the specified community (comm_index) from the community_dict. This creates a tuple of nodes belonging to the community.
    comm_nodes = tuple(u for u in community_dict \
                  if community_dict[u] == comm_index)
    #Compute the modularity matrix specific to the nodes in comm_nodes using the get_mod_matrix function. This reduces the modularity matrix to focus only on the community of interest.
    B_hat_g = modularity.get_mod_matrix(network, comm_nodes, B)

    is_sparse = issparse(B_hat_g)
    #print("Is the matrix sparse?", is_sparse)
    # compute the top eigenvector u₁ and β₁
    #For small matrices (< 3 nodes), use largest_eig to compute the largest eigenvalue and its eigenvector.
    if B_hat_g.shape[0] < 3:
        beta_s, u_s = modularity.largest_eig(B_hat_g)
    else:
        #For larger matrices, use sparse.linalg.eigs to compute the largest eigenvalue and eigenvector efficiently.
        beta_s, u_s = sparse.linalg.eigs(B_hat_g, k=1, which='LR')
    #Extract the leading eigenvector (u_1) and eigenvalue (beta_1), which are used to determine if the community can be split.
    u_1 = u_s[:, 0]
    beta_1 = beta_s[0]
    # If the largest eigenvalue is positive ( beta_1 >0), the community is divisible.
    if beta_1 > 0:
        # divisible
        #Create a division vector s, where each node is assigned +1 or −1 based on the sign of its corresponding value in u_1.
        s = sparse.csc_matrix(np.asmatrix([[1 if u_1_i > 0 else -1] for u_1_i in u_1]))
        # If refine is True, call a function (improve_modularity) to adjust the division vector s to maximize modularity.
        if refine:
            improve_modularity(network, comm_nodes, s, B)
        #Compute tha change in modularity resulting from the devision 
        delta_modularity = modularity._get_delta_Q(B_hat_g, s)
        #If modularity improves:
        if delta_modularity > 0:
            #Identify nodes that belong to the first subgroup based on the division vector s
            g1_nodes = np.array([comm_nodes[i] \
                                 for i in range(u_1.shape[0]) \
                                 if s[i,0] > 0])
            #g1 = nx.subgraph(g, g1_nodes)
            #Check for Trivial splits: If all nodes remain in one group or if the subgroup is empty, the community is considered indivisible. 
            if len(g1_nodes) == len(comm_nodes) or len(g1_nodes) == 0:
                # indivisble, return None
                return None, None
            # divisible, return node list for one of the groups
            return g1_nodes, comm_nodes
    # indivisble, return None
    return None, None

# Iteratively adjusts the community division (represented by s) to maximize modularity by finding and applying the best node swaps.
def improve_modularity(network, comm_nodes, s, B):
    '''
    Fine tuning of the initial division from `_divide`
    Modify `s` inplace

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network of interest
    comm_nodes: iterable
        List of nodes for the original group
    s: np.matrix
        A matrix of node membership. Only +1/-1
    B: np.amtrix
        Modularity matrix for `network`
    '''

    # iterate until no increment of Q
    B_hat_g = modularity.get_mod_matrix(network, comm_nodes, B)
    while True:
        #Nodes yet to be considered for reassignment in this iteration.
        unmoved = list(comm_nodes)
        # node indices to be moved
        #Tracks the order of node movements during the iteration.
        node_indices = np.array([], dtype=int)
        # cumulative improvement after moving
        node_improvement = np.array([], dtype=float)
        # keep moving until none left
        #For each unmoved node, evaluate the impact of flipping its group membership.
        while len(unmoved) > 0:
            # init Q
            Q0 = modularity._get_delta_Q(B_hat_g, s)
            scores = np.zeros(len(unmoved)) #The modularity change caused by flipping each unmoved node.
            #Temporarily flip the membership of each node, compute the change in modularity, and store the results in scores.
            for k_index in range(scores.size):
                k = comm_nodes.index(unmoved[k_index])
                s[k, 0] = -s[k, 0]
                scores[k_index] = modularity._get_delta_Q(B_hat_g, s) - Q0
                s[k, 0] = -s[k, 0]
            #Identify the node that provides the largest improvement (or smallest decrease) in modularity.
            #Move the node to the other group and update the improvement tracking.
            _j = np.argmax(scores)
            j = comm_nodes.index(unmoved[_j])
            # move j, which has the largest increase or smallest decrease
            s[j, 0] = -s[j, 0]
            node_indices = np.append(node_indices, j)
            if node_improvement.size < 1:
                node_improvement = np.append(node_improvement, scores[_j])
            else:
                node_improvement = np.append(node_improvement, \
                                        node_improvement[-1]+scores[_j])
            #print len(unmoved), 'max: ', max(scores), node_improvement[-1]
            unmoved.pop(_j)
        # the biggest improvement
        #Find the configuration of s that yielded the highest modularity improvemen
        max_index = np.argmax(node_improvement)
        # change all the remaining nodes
        # which are not helping
        #Revert moves beyond this configuration to maintain the best result.
        for i in range(max_index+1, len(comm_nodes)):
            j = node_indices[i]
            s[j,0] = -s[j, 0]
        # if we swap all the nodes, it is actually doing nothing
        if max_index == len(comm_nodes) - 1:
            delta_modularity = 0
        else:
            delta_modularity = node_improvement[max_index]
        # Stop if ΔQ <= 0
        if delta_modularity <= 0:
            break
