# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx #Library for networks and graphs 
from scipy import sparse
from scipy.linalg import eig
from itertools import product

#Input: A network with nodes labeled as strings (e.g., "A", "B", "C"); A partition mapping like {"A": 0, "B": 1, "C": 0}.
#Transformation: Node labels in the network are converted to integers (e.g., 0, 1, 2). Partition mapping is updated to use integer keys (e.g., {0: 0, 1: 1, 2: 0}).
#Output: The transformed network and partition.


def transform_net_and_part(network,partition):
    '''
    Accepts an input network and a community partition (keys are nodes,
    values are community ID) and returns a version of the network and
    partition with nodes in the range 0,...,len(G.nodes())-1.  This
    lets you directly map edges to elements of the modularity matrix.

    Returns the modified network and partition.

    network - networkx graph or digraph 
    partition - dictionary mapping nodes to their respective community IDs
    '''
    #Converts all node labels in the network to integers, starting from 0. The original node labels are stored in a node attribute called "node_name"
    #This ensures a consistent numerical labelling scheme compatible with modularity matrix compuations
    network = nx.convert_node_labels_to_integers(network, first_label=0, label_attribute="node_name")
    #Retrieves the original node labels stored in the "node_name" attribute and stores them in a dictionary node_to_name. Keys are the integer label of the node and values are the original node name
    node_to_name = nx.get_node_attributes(network, 'node_name')
    # reverse the node_name dict to flip the partition
    #Creates a reverse mapping name_to_node where keys are the original node name and the values are the corresponding integer labels. 
    #This is used to adjust the partition dictionary to match the new integer labels
    name_to_node = {v:k for k,v in node_to_name.items()}
    #Intialize empty dictionary
    int_partition = {}
    #Iterate over the original partition dictionary. For each node k, the new integer node label (name_to_node[k]) is used as the key
    #The community ID (parition[k]) is kept as the value
    #Ensures te partition aligns witht he new integer-labeled network
    for k in partition:
        int_partition[name_to_node[k]] = partition[k]
    #Returns the transformed network (network) with integer-labeled ndes. Returns the updated int_partition dictionary with integer node keys
    return network,int_partition


#REVERSE_PARTITION EXPLAINED:
#INPUT: partition = {'A': 1, 'B': 2, 'C': 1,'D': 3}
#PROCESS: Start with an empty reverse_partition = {}. Iterate over partition: Node 'A' (community 1): Add to reverse_partition = {1: ['A']}.
#Node 'B' (community 2): Add to reverse_partition = {1: ['A'], 2: ['B']}.
#Node 'C' (community 1): Append to reverse_partition = {1: ['A', 'C'], 2: ['B']}.
#Node 'D' (community 3): Add to reverse_partition = {1: ['A', 'C'], 2: ['B'], 3: ['D']}.

#OUTPUT: {1: ['A', 'C'], 2: ['B'], 3: ['D']}

def reverse_partition(partition):
    '''
    Accepts an input graph partition in the form node:community_id and returns
    a dictionary of the form community_id:[node_1,node_2,...].
    partition: A dictionary mapping nodes to community IDs (node:community_id).
    Returns: A dictionary where keys are community IDs, and values are lists of nodes belonging to those communities (community_id:[node_1, node_2, ...]).
    '''
    reverse_partition = {}
    #Iterate through keys (p representing a node) in the input partition dictionary
    for p in partition:
        # Checks if the community ID (partition[p]) is already a key in reverse_partition. If True: Appends the node (p) to the list of nodes for that community ID
        if partition[p] in reverse_partition:
            reverse_partition[partition[p]].append(p)
        else:
            #Creates a new key-value pair where the key is the community ID, and the value is a list containing the current node ([p]).
            reverse_partition[partition[p]] = [p]
    return reverse_partition

#MODULARITY EXPLAINED:
#Input: A graph with nodes connected by edges; A partition dictionary mapping nodes to communities (e.g., {0: 1, 1: 1, 2: 2, 3: 2}).
#Transformation: Convert nodes to integers if necessary. Reverse the partition to group nodes by community.
#Modularity Calculation: Compute the modularity matrix Q. Identify all within-community node pairs. Sum up the contributions of those pairs and normalize.

def modularity(network, partition):
    '''
    Computes the modularity; works for Directed and Undirected Graphs, both
    unweighted and weighted.
    '''
    # put the network and partition into integer node format
    #Action: Converts the network to integer-labeled nodes (starting from 0) and adjusts the partition accordingly using the transform_net_and_part function.
    # Why: Ensures compatibility with modularity matrix computations.
    network,partition = transform_net_and_part(network,partition)
    # get the modularity matrix
    #Action: Retrieves the modularity matrix Q for the network. This matrix captures the difference between the observed and expected edge densities.
    Q = get_base_modularity_matrix(network)
    #For undirected graphs:
    if type(network) == nx.Graph:
        #The normalization factor (norm_fac) is 2 * number of edges
        norm_fac = 2.*(network.number_of_edges())
        if nx.is_weighted(network):
            # 2*0.5*sum_{ij} A_{ij}
            #If the graph is weighted, norm_fac is the sum of all edge weights
            norm_fac = nx.to_scipy_sparse_matrix(network).sum()
    #Directed graph
    elif type(network) == nx.DiGraph:
        #Normalization factor = number of edges
        norm_fac = 1.*network.number_of_edges()
        if nx.is_weighted(network):
            # sum_{ij} A_{ij}
            #If the graph is weighted, norm_fac is the sum of all edge weights
            norm_fac = nx.to_scipy_sparse_matrix(network).sum()
    else:
        print('Invalid graph type')
        raise TypeError
    # reverse the partition dictionary
    rev_part = reverse_partition(partition)
    # get the list of all within-community pairs
    #For each community in rev_part, generates all pairs of nodes within that community using itertools.product. Stores these pairs in the pairs list.
    #These pairs represent the edges that contribute to the modularity score
    pairs = []
    for p in rev_part:
        for i,j in product(rev_part[p],rev_part[p]):
            pairs.append((i,j))
    # now sum up all the appropriate values
    return sum([Q[x] for x in pairs])/norm_fac


#Modularity Matrix: Represents the difference between the observed and expected edge densities in a network. It is central to modularity-based community detection algorithms.
def get_base_modularity_matrix(network):
    '''
    Obtain the modularity matrix for the whole network.  Assumes any edge weights
    use the key 'weight' in the edge attribute.

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network of interest

    Returns
    -------
    np.matrix
        The modularity matrix for `network`

    Raises
    ------
    TypeError
        When the input `network` does not fit either nx.Graph or nx.DiGraph
    '''
    #Undirected graph 
    if type(network) == nx.Graph:
        if nx.is_weighted(network):
            #If the graph is weighted computes the modularity matrix using the nx.modularity_matrix function with the weight='weight' parameter.
            return sparse.csc_matrix(nx.modularity_matrix(network,weight='weight'))
        #For unweighted graphs, computes the modularity matrix without the weight parameter.
        return sparse.csc_matrix(nx.modularity_matrix(network))
    elif type(network) == nx.DiGraph:
        if nx.is_weighted(network):
            return sparse.csc_matrix(nx.directed_modularity_matrix(network,weight='weight'))
        return sparse.csc_matrix(nx.directed_modularity_matrix(network))
    else:
        raise TypeError('Graph type not supported. Use either nx.Graph or nx.Digraph')


def _get_delta_Q(X, a):
    '''
    Calculate the delta modularity
    .. math::
        \deltaQ = s^T \cdot \^{B_{g}} \cdot s
    .. math:: \deltaQ = s^T \cdot \^{B_{g}} \cdot s

    Parameters
    ----------
    X : np.matrix
        B_hat_g
    a : np.matrix
        s, which is the membership vector

    Returns
    -------
    float
        The corresponding :math:`\deltaQ`
    '''

    delta_Q = (a.T.dot(X)).dot(a)
    return delta_Q[0,0]


#Calculate the modularity matrix for a specific group (comm_nodes) in the network.
#comm_nodes: A list, tuple, or array of nodes in the community for which the modularity matrix is computed. If None, defaults to the entire network.
#B: The modularity matrix of the whole network, if precomputed. If None, it will be computed internally.
def get_mod_matrix(network, comm_nodes=None, B=None):
    '''
    This function computes the modularity matrix
    for a specific group in the network.
    (a.k.a., generalized modularity matrix)

    Specifically,
    .. math::
        B^g_{i,j} = B_ij - \delta_{ij} \sum_(k \in g) B_ik
        m = \abs[\Big]{E}
        B_ij = A_ij - \dfrac{k_i k_j}{2m}
        OR...
        B_ij = \(A_ij - \frac{k_i^{in} k_j^{out}}{m}

    When `comm_nodes` is None or all nodes in `network`, this reduces to :math:`B`

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network of interest
    comm_nodes : iterable (list, np.array, or tuple)
        List of nodes that defines a community
    B : np.matrix
        Modularity matrix of `network`

    Returns
    -------
    np.matrix
        The modularity of `comm_nodes` within `network`
    '''
    #If comm_nodes is not provided, compute the modularity matrix for the entire network.
    if comm_nodes is None:
        comm_nodes = list(network)
        return get_base_modularity_matrix(network)

    if B is None:
        B = get_base_modularity_matrix(network)

    # subset of mod matrix in g
    #Find the indices of the nodes in comm_nodes relative to the network's node list.
    #Extract the submatrix B_g corresponding to these indices. This submatrix is a subset of B limited to the rows and columns for comm_nodes.

    indices = [list(network).index(u) for u in comm_nodes]
    B_g = B[indices, :][:, indices]
    #print 'Type of `B_g`:', type(B_g)

    # B^g_(i,j) = B_ij - δ_ij * ∑_(k∈g) B_ik
    # i, j ∈ g
    #Create a empty matrix B_hat_g, of the same size as B_g, filled with zeros
    B_hat_g = np.zeros((len(comm_nodes), len(comm_nodes)), dtype=float)

    # ∑_(k∈g) B_ik
    B_g_rowsum = np.asarray(B_g.sum(axis=1))[:, 0]
    #For undirected graphs (nx.Graph), row and column sums are identical. For directed graphs (nx.DiGraph), compute the column sums separately.
    if type(network) == nx.Graph:
        B_g_colsum = np.copy(B_g_rowsum)
    elif type(network) == nx.DiGraph:
        B_g_colsum = np.asarray(B_g.sum(axis=0))[0, :]

    B_hat_g = B_g.toarray()
    for i in range(B_hat_g.shape[0]):
        #Update diagonal elements 
        #?
        B_hat_g[i, i] = B_g[i, i] - 0.5 * (B_g_rowsum[i] + B_g_colsum[i])
    
    #For directed graphs, symmetrize B_hat_g by adding its transpose 
    if type(network) == nx.DiGraph:
        B_hat_g = B_hat_g + B_hat_g.T

    return sparse.csc_matrix(B_hat_g)


#calculate the largest eigenvalue and its corresponding eigenvector for a given matrix A
def largest_eig(A):
    '''
        A wrapper over `scipy.linalg.eig` to produce
        largest eigval and eigvector for A when A.shape is small
    '''
    #Use the eig function from scipy.linalg to compute all eigenvalues (vals) and eigenvectors (vectors) of the matrix A.
    vals, vectors = eig(A.todense())
    #Action: Identify indices of eigenvalues that are purely real (no imaginary part). This is done by checking if the imaginary part of each eigenvalue is zero.
    real_indices = [idx for idx, val in enumerate(vals) if not bool(val.imag)]
    #Extract only the real parts of the eigenvalues corresponding to the indices in real_indices.
    # Filter the eigenvectors to retain only those corresponding to real eigenvalues.
    vals = [vals[i].real for i in range(len(real_indices))]
    vectors = [vectors[i] for i in range(len(real_indices))]
    #Action: Sort the real eigenvalues and identify the index of the largest one. np.argsort(vals)[-1] gives the position of the maximum eigenvalue.
    max_idx = np.argsort(vals)[-1]
    #Extract the largest eigenvalue and its corresponding eigenvector. Convert them to numpy arrays for compatibility. Transpose the eigenvector to ensure it has the correct shape (column vector).
    return np.asarray([vals[max_idx]]), np.asarray([vectors[max_idx]]).T
