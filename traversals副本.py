"""
A set of functions to sample nodes of a graph *without* replacements (such as BFS). Also included are the corresponding bias correction procedures from [1].

Supported sampling methods:
- Breadth First Search (BFS)
- Depth First Search (DFS)
- Forest Fire (FF)

Also included is a corresponding bias correction procedure from [1].  
It is exact for RG(pk) - a random graph model with a known (and arbitrary) node degree distribution pk. 
This model also turns out to be a very effective approximation of real-life topologies in our context.

Supported estimators: 
- estimate_pk
- estimate_values_mean_BFS
- estimate_pk_RW

Requires free NetworkX library (http://networkx.lanl.gov)

For more details and citation, please see:
[1] M. Kurant, A. Markopoulou and P. Thiran, "On the bias of Breadth First Search (BFS)", ITC 2010, extended version submitted to JSAC.
 



Example:
--------
>>> G = nx.read_edgelist('ca-CondMat.edges.gz', create_using=nx.Graph(), nodetype=int, data=False)
>>> 2.0*G.number_of_edges()/G.number_of_nodes()    ## real average node degree
8.5513270608060665
>>> S = bfs(G, 1, int(  G.number_of_nodes()*0.1))  ## collect a BFS sample covering 10% of nodes
>>> qk = degrees2pk([G.degree(v) for v in S])    ## sampled node degree distribution
>>> mean_degree(qk)                                ## sampled average node degree
20.673220973782787
>>> mean_degree(estimate_pk(qk, 0.1))              ## corrected average node degree
9.1204757422444072


For more examples try 'test_traversals()'
"""


#Author: Maciej Kurant
#
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are
#met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import networkx as nx
import numpy as np
import random


__author__ = """Maciej Kurant"""

__all__ = ['ball',
           'bfs',
           'dfs',
           'forest_fire_sample',
           'estimate_pk', 
           'estimate_pk_RW',
           'degrees2pk', 
           'mean_degree',
           'estimate_values_mean_BFS']





###############################################################
##################   AUXILIARY FUNCTIONS  #####################
###############################################################

#####################
def __parameter_check(G, size):
    '''
    Should be treated as private
    '''

    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise nx.NetworkXException("No support for multigraphs yet!")

    size = int(size)   #just in case, gets crazy when floats are given
    
    if size > G.number_of_nodes():
        raise nx.NetworkXException("Too many nodes to collect")
        
    return size
    
    
###############################################################
#######################   TRAVERSALS  #########################
###############################################################


####################
def ball(G,v,k):
    """
    ball(G, v, k)
    
    Returns all nodes not farther than k hops away from v.
    
    Parameters
    ----------
    G:  networkx.Graph
    v:  initial node
    k:  ball radius
    """
    
    V = set([v])
    for i in range(k):
        V1 = []
        for u in V:
            V1.extend(G.neighbors(u))
        V.update(set(V1))
    return V
    
    
####################
def bfs(G, v, max_len):
    """
    bfs(G, v, max_len)
    
    Returns a Breadth First Search (BFS) sample of G, initiated at node v. 
    Its length is max_len
    
    Parameters
    ----------
    G:        networkx.Graph 
    v:        initial node
    max_len:  sample length
    """
    
    max_len = __parameter_check(G, max_len)
    
    queue, enqueued = [v], set([v])
    i = 0
    while len(queue) > i:
        for u in nx.neighbors(G, queue[i]):    # should be randomized or use decorrelate_graph(G) 
        # can be speeded up! for u in G[queue[i]]:
            if u in enqueued: continue
            enqueued.add(u)
            queue.append(u)
            if len(queue) >= max_len:
                return queue                
        i += 1
    return queue


####################
def dfs(G, v, max_len):
    """
    dfs(G, v, max_len)
    
    Returns a Depth First Search (DFS) sample of G, 
    initiated at node v. Its length is max_len
    
    Parameters
    ----------
    G:        networkx.Graph
    v:        initial node
    max_len:  sample length
    """

    max_len = __parameter_check(G, max_len)
    
    sample = [v]
    queue = [v]
    enqueued = set([v])
    
    while len(queue) > 0:
        v = queue.pop()
        for u in nx.neighbors(G, v):       # should be randomized or use decorrelate_graph(G) 
            if u in enqueued: continue
            enqueued.add(u)
            queue.append(u)
            sample.append(u)
            if len(sample)>=max_len:
                return sample
    return sample


####################
def forest_fire_sample(G, v, p, max_len):
    """
    forest_fire_sample(G, v, p, max_len)
    
    Returns a simplified version of Forest Fire sample of G, 
    initiated at node v. Its length is max_len.
    
    For details, see:
    J. Leskovec, J. Kleinberg and C. Faloutsos 
    "Graphs over time: densification laws, shrinking diameters and possible explanations"
    in KDD 2005.
    
    
    Parameters
    ----------
    G:        networkx.Graph
    v:        initial node
    p:        forward burning probability
    max_len:  sample length
    """
    
    max_len = __parameter_check(G, max_len)
    
    queue, enqueued = [v], set([v])
    i = 0
    while len(queue) < max_len:
        for u in nx.neighbors(G, queue[i]):       # should be randomized or use decorrelate_graph(G) 
            if u in enqueued: continue
            if random.random()>p: continue  # failed to burn
            enqueued.add(u)
            queue.append(u)
        i += 1

        if len(queue) < i:
            raise ValueError()
        if len(queue) == i:   # forest fire died out before reaching entire graph - revive it!
            w = random.choice([u for v in queue for u in G[v] if u not in enqueued])
            enqueued.add(w)
            queue.append(w)            
        
    return queue[:max_len]






###############################################################
########## Degree bias correction procerures ##################
###############################################################


    

####################
def __qk_t(pk, t):
    '''
    __qk_t(pk, t)
    
    Auxiliary. 
    Returns expected BFS's sampled node degree distribution qk as a function of t.
    Equation (9) in [1]
    
    Parameters
    ----------
    pk:  real node degree distribution in the entire graph
    t:   abstract time used in the model, 0 <= t <= 1 
    '''
    qk = np.array([p* (1.-t**k) for k,p in enumerate(pk)])
    return qk/sum(qk)


####################
def __f_t(pk,t):
    '''
    __f_t(pk, t)
    
    Auxiliary. 
    Returns the fraction f of nodes covered by BFS that corresponds to 'time' t
    Equation (10) in [1]
    
    Parameters
    ----------
    pk:  real node degree distribution in the entire graph
    t:   abstract time used in the model, 0 <= t <= 1 
    '''
    return 1.- sum([p*(t**k)for k,p in enumerate(pk) if p>0.])

####################
def __t_f(pk, f, max_iter=20, precision=0.005):
    '''
    __t_f(pk, f_real, max_iter=20, precision=0.005)
    
    Auxiliary. 
    Returns 'time' t that corresponds to the fraction f of nodes covered by BFS.
    Uses a very simple numerical method.
    Terminates when reaches either the target precision 
    or the maximal number of iterations max_iter.
    
    Parameters
    ----------
    pk:       real node degree distribution in the entire graph
    f:        fraction of nodes covered by BFS    
    '''
    
    T = np.array([[0.001,0], [0.999999999,0]])
    
    T[0,1] = __f_t(pk, T[0,0])
    T[1,1] = __f_t(pk, T[1,0])
    
    for i in range(max_iter):
        t_new = 0.5*T[0,0]+0.5*T[1,0]
        f_new = __f_t(pk, t_new)
            
        ii = f > f_new
        T[ii,0] = t_new
        T[ii,1] = f_new
        
        if abs(f-f_new)/f < precision:
            break

    if np.abs(T[0,1]-f) < np.abs(T[1,1]-f):
        return T[0,0]
    else:
        return T[1,0]

####################
def qk_f(pk, f, max_iter=20, precision=0.005):
    '''
    qk_f(pk, t)
    
    Returns expected BFS's sampled node degree distribution qk as a function of f.
    Equation (11) in [1]
    
    Uses a very simple numerical method.
    Terminates when reaches either the target precision 
    or the maximal number of iterations max_iter.
    
    Parameters
    ----------
    pk:  real node degree distribution in the entire graph
    f:   fraction of nodes covered by BFS   
    '''
    
    return __qk_t( pk,  __t_f(pk, f, max_iter, precision) )
    
    
####################
def __estimate_pk_t(qk, t):
    '''
    __estimate_pk_t(qk, t)
    
    Auxiliary. 
    Estimates and returns the real node degree distribution pk in the entire graph.

    Parameters
    ----------
    qk:  sampled node degree distribution
    t:   abstract time used in the model, 0 <= t <= 1 
    '''
    pk_est = np.array([q/(1.-t**k) for k,q in enumerate(qk)])
    pk_est[0]=0.
    pk_est/=sum(pk_est)
    return pk_est

    
####################
def estimate_pk(qk, f, max_iter=20, precision=0.005):
    '''
    estimate_pk(qk, f, max_iter=20, precision=0.005)
    
    Estimates and returns the real node degree distribution pk in the entire graph.
    Follows Equation (19) and a simple numerical method described in [1].
    Terminates when reaches either the target precision 
    or the maximal number of iterations max_iter.
    
    
    Parameters
    ----------
    qk:       sampled node degree distribution
    f:        fraction of nodes covered by BFS    
    '''
    
    T = np.array([[0.001,0], [0.999999999,0]])
    
    T[0,1] = __f_t( __estimate_pk_t(qk, T[0,0]), T[0,0] )
    T[1,1] = __f_t( __estimate_pk_t(qk, T[1,0]), T[1,0] )
    
    for i in range(max_iter):
        t_new = 0.25*T[0,0]+0.75*T[1,0]
        f_new = __f_t( __estimate_pk_t(qk, t_new), t_new )
            
        ii = f > f_new
        T[ii,0] = t_new
        T[ii,1] = f_new
        
        if abs(f-f_new)/f < precision:
            break
        
    if np.abs(T[0,1]-f) < np.abs(T[1,1]-f):
        return __estimate_pk_t(qk, T[0,0])
    else:
        return __estimate_pk_t(qk, T[1,0])


############################
def estimate_pk_RW(qk):
    '''
    estimate_pk_RW(qk)
    
    Estimates and returns the real node degree distribution pk, 
    assuming that the sample was collected by Random Walk (rather than BFS).
    Follows Equation (15) from [1].   
    
    Parameters
    ----------
    qk:   sampled node degree distribution
    '''
    pk_est = [q/i for i,q in enumerate(qk) if 0<i]
    pk_est.insert(0,0)
    pk_est/=sum(pk_est)
    return pk_est

############################
def degrees2pk(D):
    """
    degrees2pk(D)
    
    Interprets L as 
    """
    pk = np.bincount(D).astype(float)
    return pk/sum(pk)

############################    
def mean_degree(pk):
    return sum(i*p for i,p in enumerate(pk))
    


####################
def estimate_values_mean_BFS(G, values, sample, f):
    """
    estimate_values_mean_BFS(G, values, sample, f)
    
    Every node v has some value values[v] attached to it. 
    Based on a BFS sample of nodes that covers fraction f of nodes, 
    we estimate and return the average value over all nodes. 

    Parameters
    ----------  
    G:                 - networkx.Graph 
    values:            - dictionary: nodes -> values
    sample:            - a BFS sample of nodes
    f:                 - fraction of nodes covered by BFS  
    
    Example:
    -------
    >>> sample = bfs(G, 1, int(G.number_of_nodes()*0.1))
    >>> estimate_values_mean_BFS(G, G.degree(), sample, 0.1)
    9.1204757422443556   ## the estimation of the average node degree
    """

    if type(G) != nx.Graph:
        raise nx.NetworkXException("G must be a simple undirected graph!")     
    
    qk = degrees2pk([G.degree(v) for v in sample])     ## sampled node degree distribution
    pk_est = estimate_pk(qk, f)                          ## corrected average node degree
    W = qk/pk_est                                        ## sampling weights

    return sum(1.*values[v]/W[G.degree(v)] for v in sample) / sum(1./W[G.degree(v)] for v in sample)
        



###############################################################
###################   EXAMPLE AND TEST  #######################
###############################################################


############################    
def test_traversals():

    name = 'ca-CondMat.edges.gz'
    G = nx.read_edgelist(name, comments='#', create_using=nx.Graph(), nodetype=int, data=False, edgetype=None)
    G.name = name
    print nx.info(G)
    
    N = G.number_of_nodes()
    f = 0.1
    
    pk_real = degrees2pk([G.degree(v) for v in G])   # real node degree distribution
    qk_expected = qk_f(pk_real, f)         # expected node degree distribution in a BFS sample of fraction f 
    
    start_node =  random.choice(G.nodes())
    S = bfs(G, start_node, int(N*f))
    qk_sampled = degrees2pk([G.degree(v) for v in S])
    pk_corrected = estimate_pk(qk_sampled, f)
    
    print """
%2.1f  - mean degree in raw BFS, actually sampled
%2.1f  - mean degree in raw BFS, expected by the RG(pk) model.
----
%2.1f  - mean degree in G, calculated from the BFS sample according to the RG(pk) model.
%2.1f  - real mean degree in G
""" % (mean_degree(qk_sampled), mean_degree(qk_expected), mean_degree(pk_corrected),  mean_degree(pk_real))
    



