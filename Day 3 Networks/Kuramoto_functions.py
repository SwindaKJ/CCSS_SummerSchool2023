## This python file contains all functions necessary to simulate the Kuramoto systems, compute the order parameter and show figures of the system state during simulation.

## The main function to call is the function Kuramoto_simulator.

## LOAD PACKAGES
import numpy as np
import matplotlib
matplotlib.use('nbagg')
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib import pyplot as plt
from IPython.display import display, clear_output
import networkx as nx



def thetadot(theta, omega, K, A):
# The right-hand side of the kuramota ODE
    N = len(theta)
    dthetadt = np.zeros(N)
    
    #for i in range(N):
    #    dthetadt[i] += omega[i]
    #    for j in range(N):
    #        if A[i,j] == 1:
    #            dthetadt[i] += K / N * np.sin(theta[j]-theta[i])
    
    for i in range(N):
        dthetadt[i] = omega[i] + K / N * np.sum( A[:,i] * np.sin( theta - theta[i] ) )
    
    return dthetadt

def rungekutta(thetaOld, omega, K, A, dt):
# Perform runge Kutta 4 method to do the ODE evolution
    k1 = thetadot(thetaOld, omega, K, A)
    k2 = thetadot(thetaOld + dt/2 * k1, omega, K, A)
    k3 = thetadot(thetaOld + dt/2 * k2, omega, K, A)
    k4 = thetadot(thetaOld + dt/2 * k3, omega, K, A)
    
    thetaNew = thetaOld + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return thetaNew




def Kuramoto_simulator(par,sim_set):
    # THE MAIN FUNCTION THAT PERFORMS THE SIMULATION AND CALLS THE RIGHT FUNCTIONS FOR TIMESTEP AND VISUALISATIONS
    
    ## Obtain parameters
    N = par.N
    K = par.K
    
    A = par.A
    orderParameter = par.orderParameter
    
    updatePlot = sim_set.updatePlot
    plotEvolutions = sim_set.plotEvolution
    
    ## Initialize plot
    if updatePlot:
        plt.ion()
        fig_graph, ax_graph = plt.subplots(1,1,figsize = (7.5,7.5))
        G = nx.from_numpy_matrix(A)
        poss = nx.spring_layout(G)
        graph_plot = nx.draw_networkx(G, pos = poss, ax = ax_graph, node_size = 50, width = 0.5, with_labels = False, alpha = 0.5)
        xPositions = []
        yPositions = []
        for P in poss.values():
            xPositions.append(P[0])
            yPositions.append(P[1])
        metronomePlot, = ax_graph.plot(xPositions, yPositions, marker = '*', c = 'r', ls = '')
        
        
    
    ## Solve the kuramoto differential equation
    T = sim_set.T
    dt = sim_set.dt
    times = np.arange(0,T,dt) # in MATLAB 0:dt:T
    
    # initial conditions
    omega = np.random.rand(N) # uniform distribution between 0 and 1 for frequencies
    theta0 = np.random.normal(0,1,N) # random normal initial conditions
    theta = theta0
    
    # for saving
    thetas = np.empty((N, len(times)))
    thetas[:] = np.NaN
    orderPars = np.empty(len(times))
    orderPars[:] = np.NaN
    
    for j in range(len(times)):
        theta = rungekutta(theta, omega, K, A, dt)
        
        if updatePlot and j%sim_set.plot_interval == 0:
            metronomePlot.set_data(xPositions + 0.1 * np.cos(theta),yPositions + 0.1 * np.sin(theta))
            fig_graph.canvas.draw()
            fig_graph.canvas.flush_events()
        
        thetas[:,j] = theta;
        orderPars[j] = orderParameter(theta)
        
    ## Plotting after the simulation
    if plotEvolutions:
        figs, axs = plt.subplots(1,2, figsize = (10,5))
        ax_thetas = axs[0]
        ax_orderpar = axs[1]
        for k in np.linspace(0, len(thetas)-1, 20, dtype='int'):
            ax_thetas.plot(times, thetas[k,:]%(2*np.pi))
        ax_thetas.set_xlabel('t')
        ax_thetas.set_ylabel('theta_i (mod 2 pi)')
        
        ax_orderpar.plot(times, orderPars)
        ax_orderpar.set_xlabel('t')
        ax_orderpar.set_ylabel('order parameter r')
        
    return [thetas, orderPars]
        
        
    
    