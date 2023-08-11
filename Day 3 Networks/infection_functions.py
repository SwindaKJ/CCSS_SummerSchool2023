## This python file contains all functions necessary to simulate the infection spreading on a network.

## The main function to call is the function infection_simulator.

## LOAD PACKAGES
import numpy as np
import matplotlib
matplotlib.use('nbagg')
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib import pyplot as plt
from IPython.display import display, clear_output
import networkx as nx


def ind2sub(N,M,id):
    # Computes the row and column values of the id-th entry in the matrix
    x = 0
    y = 0
    while id / N >= 1:
        y += 1
        id -= N
    x = id
    return [x,y]
    
    
def sub2ind(N,M,x,y):
    # Computes which entry number is the entry at location x,y in the matrix
    id = y * N + x
    return id


def construct_lattice_network(N,M1,M2):
    # Constructs a lattice network (fails if N is not a square number)
    
    # Construct adjecency matrix
    A = np.zeros((N,N))
    for id in range(N):
        [i,j] = ind2sub(M2,M1,id)
        
        # Left neighbour
        if i==0:
            k = M2-1
        else:
            k = i - 1
        A[id,sub2ind(M2,M1, k,j)] = 1
        
        # Right neighbour
        if i == M2-1:
            k = 0
        else:
            k = i+1
        A[id,sub2ind(M2,M1,k,j)] = 1
        
        # Top neighbour
        if j==0:
            l = M1-1
        else:
            l = j - 1
        A[id,sub2ind(M2,M1,i,l)] = 1
        
        # Bottom neighbour
        if j == M1-1:
            l = 0
        else:
            l = j + 1
        A[id,sub2ind(M2,M1,i,l)] = 1
        
    return A
    
def construct_random_network(N, M1, M2, q):
    # Construct a random network that has average degree 4 as follows:
    # take a lattice natice and with probability q delete an edge and add another edge that is connected to one of the nodes previously connected to the deleted edge
    
    A_lat = construct_lattice_network(N, M1, M2)
    A = A_lat
    
    for id1 in range(N):
        for id2 in range(id1): # Only interested in unique combinations
            if A[id1,id2] == 1: # if existing link
                if np.random.rand() < q: # with probability q
                    # Delete edge
                    A[id1,id2] = 0
                    A[id2,id1] = 0
                    # And create another edge
                    nodes = [id1, id2]
                    rnode1 = nodes[ np.random.randint(2) ]
                    nodes = np.argwhere( A[rnode1,:] == 0)
                    rnode2 = nodes[ np.random.randint(len(nodes)) ]
                    A[rnode1,rnode2] = 1
                    A[rnode2,rnode1] = 1
                    
    return A
    
def construct_community_network(N, M1, M2, C, q, r):
    # Construct a community of C random networks. Each random network is constructed with the procedure above using probability q. The amount of edges between each community is r
    
    # Construct C community networks
    A = np.zeros((C*N,C*N))
    for i in range(C):
        Ai = construct_random_network(N,M1,M2,q)
        A[ i * N: (i+1)*N, i * N: (i+1)*N ] = Ai
        
    A_links = np.argwhere( A == 1 ) # Deletable intra-community edges
    #breakpoint()
    # Now create r edges between each community
    for i in range(C):
        for j in range(i):
            # Create r edges between communities
            for k in range(r):
                # Take one node from each of the communities
                c1 = np.random.randint( i*N, (i+1)*N-1 )
                c2 = np.random.randint( j*N, (j+1)*N-1 )
                
                # Create link between them
                A[c1,c2] = 1
                A[c2,c1] = 1
                
                # find another edge to delete                
                random_edge_index = np.random.randint(len(A_links))
                del_edge = A_links[ random_edge_index ]
                
                i_del = del_edge[0]
                j_del = del_edge[1]
                
                A[i_del,j_del] = 0
                A[j_del,i_del] = 0
                
    return A
    


def infection_simulator(par,sim_set):
    # THE MAIN FUNCTION THAT PERFORMS THE SIMULATION AND CALLS THE RIGHT FUNCTIONS FOR TIMESTEP AND VISUALISATIONS

    # Obtain parameters
    p = par.p
    N = par.N
    
    M1 = par.M1
    M2 = par.M2
    
    q = par.q
    
    C = par.C
    r = par.r
    
    timesteps = sim_set.timesteps
    showPlot = sim_set.showPlot
    showGraph = sim_set.showGraph
    plot_interval = sim_set.plot_interval
    
    network_type = par.network_type
    
    ## Obtain & create the network structure
    # Depending on the inputted wanted type of network, construct said type
    if network_type == 'lattice':
        if not M1 * M2 == N:
            print('lattice dimensions do not match with total number of nodes -- N has been redefined to be M1*M2')
            N = M1 * M2
        A = construct_lattice_network(N,M1,M2)
    elif network_type == 'random':
        if not M1 * M2 == N:
            print('lattice dimensions do not match with total number of nodes -- N has been redefined to be M1*M2')
            N = M1 * M2
        A = construct_random_network(N,M1,M2, q)
    elif network_type == 'community':
        if not M1 * M2 == N:
            print('lattice dimensions do not match with total number of nodes -- N has been redefined to be M1*M2')
            N = M1 * M2
        A = construct_community_network(N,M1,M2, C, q, r)
    elif network_type == 'custom':
        A = par.A
    else:
        raise Exception("incorrect input for network_type.")
        
    ## Initialisation
    # We introduce an infection randomly at one of the nodes
    N = len(A) # Needed because with  C communities, number of nodes is not N
    y0 = np.zeros((N,1))
    y0[np.random.randint(N)] = 1
    
    ## Initialize plots
    if showPlot:
        plt.ion()
        figs, axs = plt.subplots(1,2, figsize = (10,5))
        
        ax_graph = axs[1]
        ax_infected = axs[0]
    
        if showGraph:
            G = nx.from_numpy_matrix(A)
            poss = nx.spring_layout(G)
            graph_plot = nx.draw_networkx(G, pos=poss,ax=ax_graph, node_size = 50, width = 0.5, with_labels = False, alpha = 0.5)    
            xPositions = []
            yPositions = []
            for P in poss.values():
                xPositions.append(P[0])
                yPositions.append(P[1])
            xPositions = np.array(xPositions)
            yPositions = np.array(yPositions)
            # highlight some parts
            highlight_plot, = ax_graph.plot([],[], marker = '*', c = 'r', ls = '')
            ax_graph.set_xticks([])
            ax_graph.set_yticks([])
            ax_graph.set_title(['timestep: 0'])
        else:
            ax_graph.remove()
        
        inf_plot, = ax_infected.plot([],[])
        ax_infected.set_xlabel('t')
        ax_infected.set_ylabel('Infected I(t)')
        ax_infected.set_ylim([0, 1.05])
        ax_infected.set_xlim([0, timesteps])
    
    ## The actual simulation
    y = y0
    Infected = np.zeros(timesteps)
    ts = np.zeros(timesteps)
    
    for t in range(timesteps):
        # Loop over all the nodes
        y_next = y;
        for id in range(N):
            # If a node is infected
            if y[id] == 1:
                # We look for all uninfected neighbours
                nb = np.argwhere( A[:,id] * (1 - y_next.flatten()) == 1 )
                
                # and infect them with probability p
                for id2 in nb:
                    y_next[id2] = int( np.random.rand() < p )
        y = y_next;
        Infected[t] = np.sum(y)
        ts[t] = t
        
        ## Update plot
        if showPlot and (t % plot_interval == 0):
            if showGraph:
                infected_nodes = np.argwhere( y == 1 )
                infected_nodes = infected_nodes[:,0]
                highlight_plot.set_data(xPositions[infected_nodes],yPositions[infected_nodes])
                
                ax_graph.set_title('timestep: ' + str(t))
            
            inf_plot.set_data(range(t), Infected[:t]/N)
            
            figs.canvas.draw()
            figs.canvas.flush_events()
        
        
    I = Infected / N # To obtain the fraction of infected people/nodes
    
    return [ts,I,A]
        
        
def NetworkProperties(A):
    # Computes several properties of the network/graph that is represented by the adjecency matrix A
    # Input is the adjecency matrix A
    # Output is the average degree and a vector of the degree of each node and a plot of the network
    
    ## Use graph structure
    
    G = nx.from_numpy_matrix(A)
    
    ## Plotting
    
    fig_plot, ax_graph = plt.subplots(1,1,figsize=(5,5))
    poss = nx.spring_layout(G)
    graph_plot = nx.draw_networkx(G, pos=poss,ax=ax_graph, node_size = 50, width = 0.5, with_labels = False, alpha = 0.5)
    
    ## Compute degree
    degs = G.degree
    
    degrees = []
    for i in degs:
        degrees.append(i[1])
    
    deg_av = np.mean(degrees)
    
    return deg_av, degrees