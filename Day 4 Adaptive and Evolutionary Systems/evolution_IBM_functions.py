## This python file contains all functions necessary to simulate an individual based model with evolution in offspring

## The main function to call is the function 'evolution_IBM_simulator'

## LOAD PACKAGES
import numpy as np
import matplotlib
matplotlib.use('nbagg')
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from IPython.display import display, clear_output

def create_food(food,food_created,par):
    # Create enough food by upping the food matrix enough times at random locations (this can generate multiple food at the same location)
    
    # If we do not keep uneaten food, we set the food matrix back to zeros every timestep
    if not par.keep_uneaten_food:
        food = 0 * food
    for i in range(food_created):
        x = np.random.randint(par.Nx);
        y = np.random.randint(par.Ny);
        food[x,y] += 1
        
    return food

def reproduce_agents(ag):
    # makes an offpsring of the agent 'ag' that inherets its properties but also has some random slight variations mimicking evoluiton
    new_ag = ag
    
    # Change motility (keep between 0 and 0.5)
    dm = (2*np.random.rand()-1)*ag[5]
    new_ag[2] = max(min(new_ag[2]+dm,0.5),0)
    # Change death rate (keep between (0.01 and 1))
    dd = (2*np.random.rand()-1)*ag[6]
    new_ag[3] = max(min(new_ag[3]+dd,1),0.01)
    # Change birth rate (keep between 0 and 1)
    dr = (2*np.random.rand()-1)*ag[7]
    new_ag[4] = max(min(new_ag[4]+dr,1),0)
    
    return new_ag
    

def perform_timestep(t, agents, food, par):
    # Perform a time step of the numerical simulation
    
    # Create food
    food_created = int(np.round(par.food_input(t, len(agents))))
    food = create_food(food, food_created, par);
    
    # stop simulation if no agents left (to save time)
    if len(agents) == 0:
        return [agents,food]

    # Loop one time over the agents array to find the grid cells in which agents reside, and count how many there are in each cell
    Nx = par.Nx;
    Ny = par.Ny;
    xs = np.ceil( agents[:,0]*Nx )-1;
    ys = np.ceil( agents[:,1]*Ny )-1;
    agents_in_cell = np.zeros((Nx,Ny))
    
    for j in range(len(agents)):
        if int(xs[j])==30 or int(ys[j])==30:
            breakpoint()
        agents_in_cell[int(xs[j]),int(ys[j])] += 1
        
    # Then loop over all the agents and track if they reproduce or die
    new_agents = []
    for j in range(len(agents)):
        x = int(xs[j])
        y = int(ys[j])
        if food[x,y] > 0: # then there is food
            # Keep agent for next time step
            new_agents.append(agents[j,:])
            # see if it reproduces. The probability for that is the agent's reproduction rate or its reproduction rate multiplied by food per agent in its cell
            repro_prob = agents[j,4] * min(1, food[x,y]/agents_in_cell[x,y] )
            if np.random.rand() < repro_prob:
                new_ag = reproduce_agents( agents[j,:] )
                new_agents.append(new_ag)
        else: # Then there is a food shortage and the agent might die
            # If probabiliyt is smaller than the death rate, it dies. That is, if it is larger, the agent is still alive the next timestep
            if np.random.rand() > agents[j,3]:
                new_agents.append(agents[j,:])
                
    # Also make the new food matrix by removing as much as food as there are agents in a cell
    food = np.maximum(food-agents_in_cell,0)
    
    new_agents = np.array(new_agents)
    # Finally, simulate random movement of on a periodic domain.
    # Random movement via direction randomly unifomr and then size of step random uniform between 0 and motility value
    N_agents = len(new_agents);
    if N_agents == 0:
        return [new_agents,food]
    theta = np.random.rand(1,N_agents)*2*np.pi;
    size = np.random.rand(1,N_agents) * new_agents[:,2];
    dx = size * np.cos(theta)
    dy = size * np.sin(theta)
    new_agents[:,0] = (new_agents[:,0] + dx[0]) % 1.0;
    new_agents[:,1] = (new_agents[:,1] + dy[0]) % 1.0;
    
    return [new_agents,food]
        


def evolution_IBM_simulator(par, sim_set):
    # THE MAIN FUNCTION THAT PERFORMS THE SIMULATION AND CALLS THE RIGHT FUNCTIONS FOR TIMESTEPS AND VISUALISATIONS
    
    ##### Obtain settings for discretisation of grid
    Nx = par.Nx;
    Ny = par.Ny;
    
    ##### Initialisation of agents
    # The agents will be stored in an array. Every row will be a different agents. The columns contain the following:
    # 0: x-position
    # 1: y-position
    # 2: m (motility)
    # 3: d (death rate)
    # 4: r (growth rate)
    # 5: evol_m
    # 6: evol_d
    # 7: evol_r
    agents = np.zeros((par.initial_size,8))
    # Create random initial positions
    agents[:,0:2] = np.random.rand( par.initial_size, 2 )
    # Put in the initial values for the behaviour parameters
    agents[:,2] = par.m
    agents[:,3] = par.d
    agents[:,4] = par.r
    agents[:,5] = par.evol_m
    agents[:,6] = par.evol_d
    agents[:,7] = par.evol_r
    
    ##### Initialisation of food grid
    # Make grid as matrix
    food = np.zeros((Nx,Ny));
    
    ##### Initialize variables to save information in
    agents_over_time = []
    AGENT_COUNT = np.empty(sim_set.timesteps)
    AGENT_COUNT[:]=np.NaN
    FOOD_COUNT = np.empty(sim_set.timesteps)
    FOOD_COUNT[:]=np.NaN
    
    ##### INITIALIZE FIGURES
    plt.ion()
    
    # Figure to keep track of count of agents and food
    if sim_set.count_plot or sim_set.domain_plot or sim_set.plot_evolution:
        figs, axs = plt.subplots(2,3, figsize=(15,8))
        ax_count_plot = axs[0,0]
        ax_domain_plot = axs[0,1]
        ax_evol_m = axs[1,0]
        ax_evol_d = axs[1,1]
        ax_evol_r = axs[1,2]
        
        axs[0,2].remove()
        if not sim_set.count_plot:
            axs[0,0].remove()
        if not sim_set.domain_plot:
            axs[0,1].remove()
        if not sim_set.plot_evolution:
            axs[1,0].remove()
            axs[1,1].remove()
    
    if sim_set.count_plot:
        #fig_count_plot = plt.figure()
        #ax_count_plot = fig_count_plot.add_subplot(111)
        
        plot_agent_count, = ax_count_plot.plot(range(sim_set.timesteps), AGENT_COUNT, 'r', label = 'total agents')
        plot_food_count, = ax_count_plot.plot(range(sim_set.timesteps), FOOD_COUNT, 'b', label = 'total uneaten food')
        ax_count_plot.set_xlabel('timestep')
        ax_count_plot.set_ylabel('count')
        ax_count_plot.legend(loc = 'upper left')
    
    # Figure to keep track of food and agents in the domain
    if sim_set.domain_plot:
        #fig_domain_plot = plt.figure()
        #ax_domain_plot = fig_domain_plot.add_subplot(111)
        
        # make colormap
        N = 10
        color_map = np.ones((N+1,4))
        color_map[:,3] = 0.5
        color_map[1:,0] = 0
        color_map[1:,1] = np.linspace(1,0,N)
        color_map[1:,2] =  0
        newcmp = ListedColormap(color_map);
        
        # Plot uneaten food
        Nx = par.Nx;
        Ny = par.Ny;
        xs_food = np.linspace(0,1.0,num=Nx+1)
        ys_food = np.linspace(0,1.0,num=Ny+1)
        
        #plot_food = ax_domain_plot.pcolormesh(xs_food, ys_food, food, cmap = newcmp, vmin = 0, vmax = N)
        plot_food = ax_domain_plot.imshow(food.transpose(), interpolation = 'nearest', origin = 'lower', extent = (0,1,0,1), vmin = 0, vmax = N, cmap = newcmp)
        
        figs.colorbar(plot_food, ax=ax_domain_plot)
        
        # Plot agents
        plot_agents, = ax_domain_plot.plot(agents[:,0],agents[:,1], marker = '1', c='r', ls='')
        
        ax_domain_plot.set_xlim([0,1])
        ax_domain_plot.set_ylim([0,1])
        ax_domain_plot.set_title('timestep: 0')
    
    # Figures to keep track ov evolution in parameters m, d and r
    if sim_set.plot_evolution:
        edges = np.linspace(0,1,100);
        
        n, bins, patches_m = ax_evol_m.hist(agents[:,2], edges, color='r', density = True)
        ax_evol_m.set_title('agents: ' + str(len(agents)), loc = 'left')
        ax_evol_m.set_xlabel('motility m')
        ax_evol_m.set_ylabel('relative occurence (%)')
        ax_evol_m.set_xlim([0,0.5])
        ax_evol_m.set_ylim([0,100])
        
        n, bins, patches_d = ax_evol_d.hist(agents[:,3], edges, color='r', density = True)
        ax_evol_d.set_title('timestep: 0')
        ax_evol_d.set_xlabel('death rate d')
        ax_evol_d.set_ylabel('relative occurence (%)')
        ax_evol_d.set_xlim([0,1])
        ax_evol_d.set_ylim([0,100])
        
        n, bins, patches_r = ax_evol_r.hist(agents[:,4], edges, color='r', density = True)
        ax_evol_r.set_title('uneaten food: ' + str(int(np.sum(np.sum(food)))), loc = 'right')
        ax_evol_r.set_xlabel('growth rate r')
        ax_evol_r.set_ylabel('relative occurence (%)')
        ax_evol_r.set_xlim([0,1])
        ax_evol_r.set_ylim([0,100])
    
    ##### PERFORM TIMESTEP
    for t in range(sim_set.timesteps):
        # time step
        [agents,food] = perform_timestep(t, agents, food, par)
        
        # saving
        agents_over_time.append(agents);
        AGENT_COUNT[t]=len(agents);
        FOOD_COUNT[t]=np.sum(np.sum(food));
        
        
        # update plots
        if t%sim_set.plot_interval == 0:
            
            if sim_set.count_plot:
                plot_agent_count.set_data(range(sim_set.timesteps),AGENT_COUNT)
                plot_food_count.set_data(range(sim_set.timesteps),FOOD_COUNT)
                ax_count_plot.relim()
                ax_count_plot.autoscale_view()
                
            if sim_set.domain_plot:
                plot_food.set_data(food.transpose())
                if not len(agents) == 0:
                    plot_agents.set_data(agents[:,0],agents[:,1])
                else:
                    plot_agents.set_data([],[])
                ax_domain_plot.set_title('timestep: ' + str(t))
                
            if sim_set.plot_evolution:
                
                m_height = np.histogram(agents[:,2], edges, density = True)            
                for rect, h in zip(patches_m, m_height[0]):
                    rect.set_height(h)
                
                d_height = np.histogram(agents[:,3], edges, density = True)            
                for rect, h in zip(patches_d, d_height[0]):
                    rect.set_height(h)
                
                r_height = np.histogram(agents[:,4], edges, density = True)            
                for rect, h in zip(patches_r, r_height[0]):
                    rect.set_height(h)
                                    
                ax_evol_m.set_title('agents: ' + str(len(agents)), loc = 'left')
                ax_evol_d.set_title('timestep: ' + str(t))
                ax_evol_r.set_title('uneaten food: ' + str(int(np.sum(np.sum(food)))), loc = 'right')
                
            if sim_set.count_plot or sim_set.domain_plot or sim_set.plot_evolution:
                #display(figs)
                #clear_output(wait=True)
                figs.canvas.draw()
                figs.canvas.flush_events()
                    
    return [agents_over_time, AGENT_COUNT, FOOD_COUNT]