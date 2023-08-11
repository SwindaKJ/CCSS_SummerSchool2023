## This python file contains all functions necessary to simulate the Cellular Automoata systems, and show figures of the system state during simulation.

## The main function to call is the function CA_simulator.

## LOAD PACKAGES
import numpy as np
import matplotlib
matplotlib.use('nbagg')
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib import pyplot as plt
from IPython.display import display, clear_output


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

def CA_simulator(par,sim_set):
    # THE MAIN FUNCTION THAT PERFORMS THE SIMULATION AND CALLS THE RIGHT FUNCTIONS FOR TIMESTEP AND VISUALISATIONS

    # Obtain parameters
    N = par.N
    M = par.M
    p = par.p
    
    CA_evolution = par.CA_evolution # The evolution rule
    
    R_a = par.R_a
    R_i = par.R_i
    w_a = par.w_a
    w_i = par.w_i
    
    timesteps = sim_set.timesteps

    ## Initialisation of the grid
    A0 = ( np.random.rand(N,M) < p )
    A0 = A0.astype(int)
    
    ## Pre-calculation of the neighbours within certain radius (this makes the simulation faster)

    
    # Build proximity matrices
    prox_act = []
    prox_inh = []
    for dx in np.arange(0,max(R_i,R_a),1, dtype ='int'):
        for dy in np.arange(0,max(R_i,R_a),1, dtype = 'int'):
            dist = dx*dx+dy*dy
            if dist < R_a*R_a:
                prox_act.append( (dx,dy) )
                prox_act.append( (dx,-dy) )
                prox_act.append( (-dx,dy) )
                prox_act.append( (-dx,-dy) )
            elif dist < R_i * R_i:
                prox_inh.append( (dx,dy) )
                prox_inh.append( (dx,-dy) )
                prox_inh.append( (-dx,dy) )
                prox_inh.append( (-dx,-dy) )
    
    # Then use it to find all neighbours (negative values are fine!)
    neighbours_a = []
    neighbours_i = []
    perc_fin = 0
    
    for i in range(N*M):
        [xi, yi] = ind2sub(N,M,i)
        NN_a = set()
        NN_i = set()
        for pos in prox_act:
            x = ( xi + pos[0] ) % M
            y = ( yi + pos[1] ) % N
            NN_a.add( (y,x) )
        for pos in prox_inh:
            x = ( xi + pos[0] ) % M
            y = ( yi + pos[1] ) % N
            NN_i.add( (y,x) )
        # remove points in NN_i that are also in NN_a
        NN_i = NN_i - NN_a
        # Then convert to the right format (x1,x2,x3,...), (y1,y2,y3,...)
        xs_a = []
        ys_a = []
        for pos in NN_a:
            xs_a.append(pos[1])
            ys_a.append(pos[0])
        neighbours_a.append( ( tuple(ys_a), tuple(xs_a) ) )
        
        xs_i = []
        ys_i = []
        for pos in NN_i:
            xs_i.append(pos[1])
            ys_i.append(pos[0])
        neighbours_i.append( ( tuple(ys_i), tuple(xs_i) ) )


    ## Initialisation of plot
    if sim_set.show_domainplot:
        plt.ion()
        fig_domain, ax_domain = plt.subplots(1,1,figsize=(5,5))
        plot_domain = ax_domain.imshow(A0.transpose(), interpolation = 'nearest', origin = 'lower', extent = (0,M,0,N), vmin = 0, vmax = 1, cmap = 'Greys', animated = True)
        ax_domain.set_xlabel('x')
        ax_domain.set_ylabel('y')
        ax_domain.set_title('timestep: 0')

    ## Actual simulation
    A = A0;
    
    Anext = np.zeros((N,M))
    for t in range(timesteps):
        # Loop over cells and apply evolution rules
        for i in range(M):
            for k in range(N):
                id = sub2ind(N,M,k,i)
                Anext[k,i] = CA_evolution(i,k,np.array(A),neighbours_a[id],neighbours_i[id], w_a, w_i)
                
        A = Anext
        
        # Update plotting
        if sim_set.show_domainplot and ( t%sim_set.plot_interval == 0 ):
            plot_domain.set_data(A.transpose())
            ax_domain.set_title('timestep: ' + str(t))
            
            fig_domain.canvas.draw()
            fig_domain.canvas.flush_events()
            
    
    return A
        
    
    