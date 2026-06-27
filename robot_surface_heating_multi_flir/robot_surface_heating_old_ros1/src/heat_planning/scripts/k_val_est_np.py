import itertools
import time
from functools import partial, partialmethod
import jax
import jax.numpy as np
import jax.ops
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import display
from jax import flatten_util, jit, random, value_and_grad, vmap, config
# from jax.config import config
# from jaxopt import ScipyBoundedMinimize, ScipyMinimize
from sklearn.metrics import mean_absolute_error, mean_squared_error
import csv
import yaml
import math
import os
from jax import lax
from scipy.optimize import minimize


# heated_node = 29
if False:
    param_file = open('src/plot_info.yaml')
    params = yaml.full_load(param_file)

    row_count = params['row_count']
    col_count = params['col_count']

    config.update("jax_enable_x64", True)

    # dataPath = "data/flat_3_no_cooling.csv"
    # dataPath1 = "data/Training_Data/fullheating.csv"
    # dataPath = params['raw_data_file_path']
    # dataPath3 = "data/Training_Data/node5.csv"
    # dataPath4 = "data/Training_Data/node7.csv"

    # testdataPath1 = "data/Testing_Data/node5.csv"
    # testdataPath = "data/Testing_Data/node6.csv"
    # testdataPath3 = "data/Testing_Data/node9.csv"
    dataPath = params['temp_data_file_path']
    k_value_data_file_path = params['k_value_data_file_path']
    curve_fit_plot_file_path = params['curve_fit_plot_file_path']

# DATAPATH = dataPath3


def ensure_directories_exist(file_paths):
    for file_path in file_paths:
        directory = os.path.dirname(file_path)

        # Check if the directory exists
        if not os.path.exists(directory):
            # If the directory doesn't exist, create it
            os.makedirs(directory)


class FEAProblem():
    
    
    
    def __init__(self, horizon = 10, dt = 0.1, model_sheet_params = None):
        """FEAProblem class
           
        Args:
        
            nodes (array-like): node numbers that should match data indices
            left_bc (real): Dirchelet boundary on the left of the domain 
            right_bc (real): Dirchelet boundary on the right of the domain 
        """
        if model_sheet_params is None:
            self.row_len = row_count
            self.col_len = col_count
        else:
            self.row_len = model_sheet_params['row_count']
            self.col_len = model_sheet_params['col_count']
        self.nodes = np.zeros((self.row_len, self.col_len), dtype='float64')
        self.create_nx_graph(1)
        self.horizon = horizon
        self.dt = dt

        
        return
    
    def set_horizon_dt(self, horizon, dt):
        self.horizon = horizon
        self.dt = dt
        self.num_steps = int(horizon/dt)
    
    def create_nx_graph(self, layers):
    # Create an nx graph that represents the mesh , input should be size of mesh of interest (n by n) and the number of layers in the mesh
        # The mesh is a square grid with n nodes on each side it should be a quadrilateral mesh


        node_layers = []
        boundary_layers = []
        all_layers = []


        # Create heat node
        G = nx.Graph()
        G.add_node((0,0,0), temperature = 130)
        boundary_layers.append(G)
        all_layers.append(G)

        m = self.row_len
        n = self.col_len
        

        # Create the internal layers
        for layer in range(layers):
            G = nx.grid_2d_graph(m, n)
            # did not record temperature for boundary corner nodes, we will need to disable this for new data
            # G.remove_node((0, 0))
            # G.remove_node((0, n-1))
            # G.remove_node((m-1, 0))
            # G.remove_node((m-1, n-1))

            # Relabel nodes based on what layer they are in
            mapping = {(x, y): (x, y, layer+1) for x, y in G.nodes()}
            G = nx.relabel_nodes(G, mapping)

            # add temperature attribute to node # This is not being used right now, we get this temperature from initial condition
            # for node in G.nodes():
            #     G.nodes[node]['temperature'] = 70
            node_layers.append(G)
            all_layers.append(G)

        # # create boundary layer
        
        G = nx.grid_2d_graph(m, n)
        # did not record temperature for boundary corner nodes, we will need to disable this for new data
        # G.remove_node((0, 0))
        # G.remove_node((0, n-1))
        # G.remove_node((m-1, 0))
        # G.remove_node((m-1, n-1))

        # Relabel nodes based on what layer they are in
        mapping = {(x, y): (x, y, layers+1) for x, y in G.nodes()}
        G = nx.relabel_nodes(G, mapping)

        # for node in G.nodes():
        #     G.nodes[node]['temperature'] = 70
        boundary_layers.append(G)
        all_layers.append(G)

        # Combine all layers to create a mesh
        mesh = nx.union_all(all_layers)

        # Connect the layers so that each node in layer i is connected to the same node in layer i+1
        for i in range(len(all_layers) - 1):
            for (x, y, _) in all_layers[i+1].nodes():
                # Add an edge between the same node in layer i and layer i+1
                if i == 0:
                    mesh.add_edge((0, 0, 0), (x, y, i+1))
                else:
                    mesh.add_edge((x, y, i), (x, y, i+1))

        self.mesh = mesh    

        self.boundary_nodes = list(itertools.chain.from_iterable(G.nodes() for G in boundary_layers))
        self.internal_nodes = list(itertools.chain.from_iterable(G.nodes() for G in node_layers))



    # def setup_k_mat(self):
    #     """Integrates the element stiffness matrices for linear basis functions
    #        with 2 points Guass integration
    #     """
        
    #     nodes = self.nodes
    #     mesh = self.mesh



    #     # create conductivity coeff variable for each edge in graph as dictionary
    #     conductivity_dict = {tuple(sorted(list(edge))): .01 for edge in self.mesh.edges()}

    #     # initialize k matrix. 
    #     k_mat = np.zeros( (len(self.internal_nodes),len(self.mesh.nodes())), dtype='float64')


    #     for node in self.internal_nodes:
    #         # get node neighbors
    #         neighbors = list(self.mesh.neighbors(node))

    #         node_cum_k = 0
    #         # Create a direction-independent edge ID
    #         for neighbor in neighbors:
    #             edge_id = tuple(sorted((node, neighbor)))
    #             # Get the conductivity value using the edge ID
    #             conductivity = conductivity_dict.get(edge_id)
    #             node_cum_k -= conductivity

    #             # get node index in self.internal_nodes
    #             node_index = self.internal_nodes.index(node)
    #             if neighbor in self.internal_nodes:
    #                 neighbor_index = self.internal_nodes.index(neighbor)
    #             else:
    #                 neighbor_index = self.boundary_nodes.index(neighbor) + len(self.internal_nodes)
    #             k_mat = k_mat.at[node_index, neighbor_index].set(conductivity)
    #         k_mat = k_mat.at[node_index, node_index].set(node_cum_k)

    #     return k_mat

    # def update_k_mat(self,conductivity_dict):
    #     # initialize k matrix. 
    #     k_mat = np.zeros( (len(self.internal_nodes),len(self.mesh.nodes())), dtype='float64')
    #     for node in self.internal_nodes:
    #         # get node neighbors
    #         neighbors = list(self.mesh.neighbors(node))

    #         node_cum_k = 0
    #         # Create a direction-independent edge ID
    #         for neighbor in neighbors:
    #             edge_id = tuple(sorted((node, neighbor)))
    #             # Get the conductivity value using the edge ID
    #             conductivity = conductivity_dict.get(edge_id)
    #             node_cum_k -= conductivity

    #             # get node index in self.internal_nodes
    #             node_index = self.internal_nodes.index(node)
    #             if neighbor in self.internal_nodes:
    #                 neighbor_index = self.internal_nodes.index(neighbor)
    #             else:
    #                 neighbor_index = self.boundary_nodes.index(neighbor) + len(self.internal_nodes)
    #             k_mat = k_mat.at[node_index, neighbor_index].set(conductivity)
    #         k_mat = k_mat.at[node_index, node_index].set(node_cum_k)
        
    #     return k_mat

   


    # @partial(jit, static_argnums=(0,))
    # def solve(self):
    #     """ Solve ode via eulers method
        
    #     Args:
    #         tolerance (float): the tolerence at which the Newton-Raphson 
    #                            iteration stops
        
    #     Returns:
    #         Time series prediction of heated nodes
    #     """

    #     dt = self.dt
    #     horizon = self.horizon
        
    #     # Integrate the shape functions over each element
    #     k_mat = self.setup_k_mat()
        

    
    #     node_temp = np.zeros(len(self.internal_nodes), dtype='float64')
    #     boundary_temp = np.zeros(len(self.boundary_nodes), dtype='float64')

    #     # record the temperature for each time step
    #     print(self.num_steps)
    #     input()
    #     p = np.zeros((self.num_steps, len(self.internal_nodes)), dtype='float64')

    #     # initial conditions from mesh node attributes
    #     for node in self.mesh.nodes():
    #         if node in self.internal_nodes:
    #             node_temp = node_temp.at[self.internal_nodes.index(node)].set(self.mesh.nodes[node]['temperature'])
    #         else:
    #             boundary_temp = boundary_temp.at[self.boundary_nodes.index(node)].set(self.mesh.nodes[node]['temperature'])

    #     # use eulers method to solve the ode using horizon and dt as time step boundary temperature does not change
    #     for t in range(self.num_steps):
    #         # update internal nodes
    #         catenat = np.hstack((node_temp, boundary_temp))
    #         print(catenat.shape)
    #         print(k_mat.shape)

    #         node_temp = node_temp + dt * k_mat @ catenat
    #         # update boundary nodes
    #         # boundary_temp = boundary_temp + self.dt * k_mat @ np.hstack(node_temp ,boundary_temp)
    #         # record temperature
    #         p = p.at[t].set(node_temp)



                

    #     # Return the solution
    #     return p
    

    def get_nodes(self):
        # return internal and boundary nodes
        return self.internal_nodes, self.boundary_nodes
    
    def get_neighbors_list(self):   
        # return a dictionary of neighbors for each node 
        return {node: list(self.mesh.neighbors(node)) for node in self.mesh.nodes}
    
    def get_edges(self):
        # return edges of mesh
        return self.mesh.edges()



class ThermModel():
    def __init__(self,fea,model_sheet_params=None):

        internal_nodes, boundary_nodes = fea.get_nodes()
        neighbors_list = fea.get_neighbors_list()
        edges = list(fea.get_edges())

        self.internal_nodes = internal_nodes
        self.boundary_nodes = boundary_nodes
        self.neighbors_list = neighbors_list
        self.edges = edges
        self.precompute_edge_indices() # precompute edge indices for faster computation
        self.data_to_graph_mapping = {}

        row = model_sheet_params['row_count']
        col = model_sheet_params['col_count']
        self.row = row
        self.col = col

        curr_node_no = 1

        for i in range(row):
            for j in range(col):
                self.data_to_graph_mapping[curr_node_no] = (i, j ,1)
                curr_node_no+=1
        
        

        # create graph to data mapping
        self.graph_to_data_mapping = {v: k for k, v in self.data_to_graph_mapping.items()}
            ## Initialize the distance matrix
        distance_matrix = np.zeros((len(internal_nodes), len(internal_nodes)))
        # Calculate the distance between each pair of nodes
        for i in range(len(internal_nodes)):
            for j in range(len(internal_nodes)):
                node_i = self.data_to_graph_mapping[i+1]
                node_j = self.data_to_graph_mapping[j+1]
                distance_matrix[i][j] = np.sqrt((node_i[0] - node_j[0])**2 + (node_i[1] - node_j[1])**2)

        self.distance_matrix = distance_matrix

        node_tup_list = []
        for i in range(len(internal_nodes)):
            node_tup_list.append(self.data_to_graph_mapping[i+1])

        self.node_tup_list = node_tup_list
        
        ######################### Other model attributes ############################
        if model_sheet_params is not None:
            self.model_parameters = model_sheet_params['model_params']
        else:
            self.model_parameters = None
        self.cached_k_matrix = {}

    def precompute_edge_indices(self):
        # precompute edge indices that will be used in the simulation

        edges = self.edges

        internal_nodes = self.internal_nodes
        boundary_nodes = self.boundary_nodes

        # create empty list the size of all nodes
        edge_indices = [None] * len(edges)

        # for each edge we should have node indices 
        for edge in edges:
            edge_index = edges.index(edge)
            node = edge[0]
            neighbor = edge[1]

            # get the index of the node in the internal nodes
            if node in internal_nodes:
                node_index = internal_nodes.index(node)
            else:
                node_index = boundary_nodes.index(node) + len(internal_nodes)
            
            if neighbor in internal_nodes:
                neighbor_index = internal_nodes.index(neighbor)
            else:
                neighbor_index = boundary_nodes.index(neighbor) + len(internal_nodes)
        

            edge_indices[edge_index] = (node_index, neighbor_index)
        self.edge_indices = edge_indices

    # @partial(jit, static_argnums=(0,))
    def learn_model_parameters(self):

        internal_nodes = self.internal_nodes
        boundary_nodes = self.boundary_nodes
        neighbors_list = self.neighbors_list

        neighbors_keys = list(neighbors_list.keys())
        neighbors_keys = np.array(neighbors_keys)

        edges = self.edges

        horizon = self.data['time'].iloc[-1]
          
        dt = 0.1


        node_tup_list = self.node_tup_list
        # initialize k values vector of .1 with same length as number of edges
        # model_params_init = np.ones(len(edges)) * .1

        # lower_bounds = np.zeros_like(model_params_init)
        # upper_bounds = np.ones_like(model_params_init)

        # # initialize vector environmental nodes of 70 with same length as number of boundary nodes
        # model_params_init = np.hstack((model_params_init, np.ones(len(boundary_nodes)) * 90))

        # # change first boundary node to 130
        # model_params_init = model_params_init.at[len(edges)].set(400)

        # lower_bounds = np.hstack((lower_bounds, np.zeros(len(boundary_nodes))))
        # upper_bounds = np.hstack((upper_bounds, np.full(len(boundary_nodes), 800)))
        # # Add two more parameters for heat cone k value and sigma
        # model_params_init = np.hstack((model_params_init, np.array([.1, .1])))

        # lower_bounds = np.hstack((lower_bounds, np.zeros(2)))
        # upper_bounds = np.hstack((upper_bounds, np.array([20, 20])))


        model_params_init = np.array([.1,.1,400,80,.1,.1,.1,1])
        # lower_bounds = np.zeros_like(model_params_init)
        # upper_bounds = np.ones_like(model_params_init)*np.inf
        lower_bounds = np.array([0,0,0,0,-1,-1,0,0])
        upper_bounds = np.array([1,1,800,800,1,1,np.inf,np.inf])
        bounds = (lower_bounds, upper_bounds)


        # lower_bounds = np.zeros_like(model_params_init)
        # upper_bounds = np.ones_like(model_params_init)*np.inf

        bounds = (lower_bounds, upper_bounds)
        # initial conditions from first time step in data # We are dropping time and heated node column
        # data_copy = self.data.drop(columns=['time', 'heated_node'])
        data_copy = self.data.drop(columns=['time'])
        
        # data_copy = data_copy[internal_nodes]

        initial_conditions = data_copy.iloc[0, :-1].to_numpy().reshape(-1, 1)
        data = data_copy.to_numpy()

        initial_conditions = np.array(initial_conditions, dtype='float64')

      
        # Define the objective function with extra arguments
        def objective_fn(model_params, initial_conditions, horizon, dt, internal_nodes, boundary_nodes, edges, data, node_tup_list):
            return self.simulate_model(model_params, initial_conditions, horizon, dt, internal_nodes, boundary_nodes, edges, data, node_tup_list)

        # Create a partial function with the extra arguments
        objective_fn_with_args = partial(objective_fn, initial_conditions=initial_conditions, horizon=horizon, dt=dt, internal_nodes=internal_nodes, boundary_nodes=boundary_nodes, edges=edges, data=data, node_tup_list=node_tup_list)

        # Set up the optimizer with the partial function
        optimizer = ScipyBoundedMinimize(fun=objective_fn_with_args, method='L-BFGS-B',maxiter=5000)
 
        # Run the optimization with bounds
        # print(bounds)
        # input("here")


        # Set up the optimizer with the partial function and box constraints
        # lower_bounds = np.zeros(len(edges) + len(boundary_nodes))  # Lower bound of 0 for all parameters
        # upper_bounds = np.full(len(edges), .3)  # No upper bound
        # upper_bounds = np.hstack((upper_bounds, np.full(len(boundary_nodes), 800)))
        # bounds = (lower_bounds, upper_bounds)

        

        start = time.time()
        # result = optimizer.run(model_params_init, bounds=bounds)
        # print("Optimization time:", time.time() - start)

        # # Extract the optimized parameters
        # optimized_params = result.params
        # # print("Optimized Parameters:", optimized_params)
        # print()
        # print("Optimized Loss:", result.fun)
        optimized_params = [7.24099179e-02, 6.95412421e-02, 3.39597724e+02, 1.88715038e+02, 2.58761949e-01, 1.32251134e-01, 4.21859462e-02, 6.44575960e-01]

        # Simulate the model with the optimized parameters
        temp_results = self.simulate_model_eval(optimized_params, initial_conditions, horizon, dt, internal_nodes, boundary_nodes, edges, data,node_tup_list)
        print("simulated")
        temp_data = data[:, :-1]
        fit_curve_df = pd.DataFrame()

        fit_curve_df["Time_Seconds"] = np.arange(data.shape[0] - 2) * .1
        # Visualize the results
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create 2 subplots side by side
        predicted_results = temp_results
        for i in range(temp_results.shape[1]):
            time_steps = np.arange(data.shape[0] - 2) * .1
            axs[0].plot(time_steps,data[:-2, i], label=f'Location {i+1}')
            axs[1].plot(time_steps,predicted_results[:-2, i], label=f'Location {i+1}')
            fit_curve_df["Node_"+str(i)] = predicted_results[:-2, i]
            mae = mean_absolute_error(temp_data, predicted_results)
            mse = mean_squared_error(temp_data, predicted_results)

            # Add MAE and MSE to the plots
            axs[0].text(0.05, 0.95, f'MAE: {mae:.2f}\nMSE: {mse:.2f}', transform=axs[0].transAxes, verticalalignment='top')
            axs[1].text(0.05, 0.95, f'MAE: {mae:.2f}\nMSE: {mse:.2f}', transform=axs[1].transAxes, verticalalignment='top')

        fit_curve_df.to_csv(params["fit_curve_file_path"]+str(row_count)+"_"+str(col_count)+"_"+str(params['heat_node_row'])+"_"+str(params['heat_node_col'])+'_fit_curve.csv', index=False)
        axs[0].set_title('Actual Temperature Results')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Temperature')

        axs[1].set_title('Predicted Temperature Results')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Temperature')

        # Set the same x and y limits for both plots
        ylim = (70, 170)  # Replace with your actual min and max temperatures

        # axs[0].set_xlim(xlim)
        axs[0].set_ylim(ylim)

        # axs[1].set_xlim(xlim)
        axs[1].set_ylim(ylim)

        plt.tight_layout()  # Adjust layout to not overlap subplots
        plt.savefig(curve_fit_plot_file_path+str(row_count)+"x"+str(col_count)+"_"+str(params['heat_node_row'])+"_"+str(params['heat_node_col'])+'_2_subplots.png')  # Save the figure to a file

        plt.figure("matching two plots")
        for i in range(temp_results.shape[1]):
            time_steps = np.arange(data.shape[0] - 2) * .1
            plt.plot(time_steps,data[:-2, i], label=f'Location {i+1}')
            plt.plot(time_steps,predicted_results[:-2, i], label=f'Location {i+1}')

        plt.tight_layout()
        plt.savefig(curve_fit_plot_file_path+str(row_count)+"x"+str(col_count)+"_"+str(params['heat_node_row'])+"_"+str(params['heat_node_col'])+'_pred_fit_curves_match.png')
        plt.show()
        pass

    # @partial(jit, static_argnums=(0,3,4))
    def simulate_model(self,model_params,initial_conditions,horizon,dt,internal_nodes, boundary_nodes,edges,data,node_tup_list):

        x_correction_factor = model_params[4] # Define your x correction factor here
        y_correction_factor = model_params[5]  # Define your y correction factor here


        distance_matrix = np.zeros((len(internal_nodes), len(internal_nodes)))
        # Calculate the distance between each pair of nodes
        for i in range(len(internal_nodes)):
            for j in range(len(internal_nodes)):
                node_i = node_tup_list[i]
                node_j = node_tup_list[j]
                # distance_matrix[i][j] = np.sqrt((node_i[0]+x_correction_factor - node_j[0])**2 + (node_i[1] + y_correction_factor - node_j[1])**2)
                distance_matrix[i, j] = np.sqrt((node_i[0]+x_correction_factor - node_j[0])**2 + (node_i[1] + y_correction_factor - node_j[1])**2)
        
        
        def create_k_matrix(model_params, internal_nodes, boundary_nodes, edges,heated_node):
            def gaussian_cone_k(distance,sigma,k0):
                return k0 * np.exp(-distance**2/(2*sigma**2))

            k0 = model_params[-2]
            sigma = model_params[-1]

            k_mat = np.zeros( (len(internal_nodes),len(boundary_nodes)+len(internal_nodes)), dtype='float64')

            for i in range(len(self.edge_indices)):
                # k_val = model_params[i]
                node_index, neighbor_index = self.edge_indices[i]
                if node_index < len(internal_nodes):
                    if neighbor_index == len(internal_nodes): # node is connected to heater node
                        k_val = gaussian_cone_k(distance_matrix[heated_node][node_index], sigma, k0)
                        k_mat[node_index, neighbor_index] = k_val
                    elif neighbor_index < len(internal_nodes): # node is connected to another internal node
                        k_mat[node_index, neighbor_index] = model_params[0]
                    else: # node is connected to boundary node
                        k_mat[node_index, neighbor_index] = model_params[1]
                    # k_mat = k_mat.at[node_index, neighbor_index].set(k_val)
                if neighbor_index < len(internal_nodes):
                    # k_mat = k_mat.at[neighbor_index, node_index].set(k_val)
                    if(node_index == len(internal_nodes)):
                        k_val = gaussian_cone_k(distance_matrix[heated_node][neighbor_index], sigma, k0)
                        k_mat[neighbor_index, node_index] = k_val
                    elif neighbor_index < len(internal_nodes): # node is connected to another internal node
                        k_mat[neighbor_index, node_index] = model_params[0]
                    else: # node is connected to boundary node
                        k_mat[neighbor_index, node_index] = model_params[1]
                
                # add k val to diagonal
                if node_index < len(internal_nodes):
                    k_mat[node_index, node_index] = k_mat[node_index, node_index] - model_params[0]
                if neighbor_index < len(internal_nodes):
                    k_mat[neighbor_index, neighbor_index] = k_mat[neighbor_index, neighbor_index] - model_params[0]
        
            return k_mat
        
        
        
        temp_data = data[:, :-1]
        heated_nodes = data[:, -1].reshape(-1).astype(int)


        num_steps = len(data)-1

        temps = np.zeros((len(data), len(internal_nodes)), dtype='float64') # array for simulated temperature nodes

        # set first row to initial condition temperatures
        temps[0] = initial_conditions.flatten()

        # array for boundary and internal nodes temperature
        node_temp = np.zeros(len(internal_nodes), dtype='float64')
        boundary_temp = np.zeros(len(boundary_nodes), dtype='float64')

        # initial conditions 
        for node_ind in range(len(internal_nodes)):
            init_temp = initial_conditions[node_ind][0]
            node_temp[node_ind] = init_temp

        for bound_in in range(len(boundary_nodes)):
            node_ind = bound_in + len(edges)
            if node_ind == len(edges):
                boundary_temp[bound_in] = model_params[2] 
            else:
                boundary_temp[bound_in] = model_params[3]


        ################## Simulate model ##################
        prev_heated_node = None
        for t in range(1,num_steps+1):
            # update internal nodes
            current_heated_node = heated_nodes[t]

            if prev_heated_node != current_heated_node:
                k_mat = create_k_matrix(model_params, internal_nodes, boundary_nodes, edges, current_heated_node)
            prev_heated_node = current_heated_node
            catenat = np.hstack((node_temp, boundary_temp))
            node_temp = node_temp + dt * k_mat @ catenat
            # record temperature
            temps[t] = node_temp
        ####################################################
        # get mse loss between data and temps
        mse = np.sum((temp_data - temps) ** 2)

        # Get mse of 1st derivative
        d_mse = np.sum((np.diff(temp_data) - np.diff(temps)) ** 2)

        mse = mse +d_mse

        # L2 regularization term with lambda minus mean of parameters
        #  np.sum(np.mean(model_params[0:len(edges)])-model_params[0:len(edges)] ** 2)

        lambda_val = .01
        mse = mse + lambda_val * np.sum(model_params[0:len(edges)] ** 2)

        return mse
    
    def simulate_model_eval(self,model_params,initial_conditions,horizon,dt,internal_nodes, boundary_nodes,edges,data,node_tup_list):
        

        x_correction_factor = model_params[4] # Define your x correction factor here
        y_correction_factor = model_params[5]  # Define your y correction factor here


        distance_matrix = np.zeros((len(internal_nodes), len(internal_nodes)))
        # Calculate the distance between each pair of nodes
        for i in range(len(internal_nodes)):
            for j in range(len(internal_nodes)):
                node_i = node_tup_list[i]
                node_j = node_tup_list[j]
                # distance_matrix[i][j] = np.sqrt((node_i[0]+x_correction_factor - node_j[0])**2 + (node_i[1] + y_correction_factor - node_j[1])**2)
                distance_matrix[i, j] = np.sqrt((node_i[0]+x_correction_factor - node_j[0])**2 + (node_i[1] + y_correction_factor - node_j[1])**2)

        
        def create_k_matrix(model_params, internal_nodes, boundary_nodes, edges,heated_node):
            def gaussian_cone_k(distance,sigma,k0):
                return k0 * np.exp(-distance**2/(2*sigma**2))

            k0 = model_params[-2]
            sigma = model_params[-1]

            k_mat = np.zeros( (len(internal_nodes),len(boundary_nodes)+len(internal_nodes)), dtype='float64')

            for i in range(len(self.edge_indices)):
                # k_val = model_params[i]
                node_index, neighbor_index = self.edge_indices[i]
                if node_index < len(internal_nodes):
                    if neighbor_index == len(internal_nodes): # node is connected to heater node
                        k_val = gaussian_cone_k(distance_matrix[heated_node][node_index], sigma, k0)
                        k_mat[node_index, neighbor_index] = k_val
                    elif neighbor_index < len(internal_nodes): # node is connected to another internal node
                        k_mat[node_index, neighbor_index] = model_params[0]
                    else: # node is connected to boundary node
                        k_mat[node_index, neighbor_index] = model_params[1]
                    # k_mat = k_mat.at[node_index, neighbor_index].set(k_val)
                if neighbor_index < len(internal_nodes):
                    # k_mat = k_mat.at[neighbor_index, node_index].set(k_val)
                    if(node_index == len(internal_nodes)):
                        k_val = gaussian_cone_k(distance_matrix[heated_node][neighbor_index], sigma, k0)
                        k_mat[neighbor_index, node_index] = k_val
                    elif neighbor_index < len(internal_nodes): # node is connected to another internal node
                        k_mat[neighbor_index, node_index] = model_params[0]
                    else: # node is connected to boundary node
                        k_mat[neighbor_index, node_index] = model_params[1]
                
                # add k val to diagonal
                if node_index < len(internal_nodes):
                    k_mat[node_index, node_index] = k_mat[node_index, node_index] - model_params[0]
                if neighbor_index < len(internal_nodes):
                    k_mat[neighbor_index, neighbor_index] = k_mat[neighbor_index, neighbor_index] - model_params[0]
        
            return k_mat
        
        temp_data = data[:, :-1]
        heated_nodes = data[:, -1].reshape(-1).astype(int)
        
        k_mat= create_k_matrix(model_params, internal_nodes, boundary_nodes, edges, heated_nodes[0])
        num_steps = len(data)-1


        # heat_to_sheet_k_val_file_path = k_value_data_file_path+str(row_count)+"x"+str(col_count)+"_"+str(params['heat_node_row'])+"_"+str(params['heat_node_col'])+"_heat_to_sheet_k_val.csv"
        # node_edge_k_val_file_path = k_value_data_file_path+str(row_count)+"x"+str(col_count)+"_"+str(params['heat_node_row'])+"_"+str(params['heat_node_col'])+"_node_edge_k_val.csv"
        
        # with open(heat_to_sheet_k_val_file_path, mode='w', newline='') as file:
        #     writer = csv.writer(file)

        #     for row in node_to_k_tuple_list:
        #         writer.writerow(row)
        
        # with open(node_edge_k_val_file_path, mode='w', newline='') as file:
        #     writer = csv.writer(file)

        #     for row in sheet_node_edge_k:
        #         writer.writerow(row)
        
        temps = np.zeros((len(data), len(internal_nodes)), dtype='float64') # array for simulated temperature nodes

        # set first row to initial condition temperatures
        temps[0] = initial_conditions.flatten()

        # array for boundary and internal nodes temperature
        node_temp = np.zeros(len(internal_nodes), dtype='float64')
        boundary_temp = np.zeros(len(boundary_nodes), dtype='float64')

        # initial conditions 
        for node_ind in range(len(internal_nodes)):
            init_temp = initial_conditions[node_ind][0]
            node_temp[node_ind] = init_temp

        for bound_in in range(len(boundary_nodes)):
            node_ind = bound_in + len(edges)
            if node_ind == len(edges):
                boundary_temp[bound_in] = model_params[2]
            else:
                boundary_temp[bound_in] = model_params[3]


        ################## Simulate model ##################
        prev_heated_node = None
        for t in range(1,num_steps+1):
            # update internal nodes
            current_heated_node = heated_nodes[t]

            if prev_heated_node != current_heated_node:
                k_mat  = create_k_matrix(model_params, internal_nodes, boundary_nodes, edges, current_heated_node)
            
            prev_heated_node = current_heated_node
            catenat = np.hstack((node_temp, boundary_temp))
            node_temp = node_temp + dt * k_mat @ catenat
            
            
            print(node_temp)
            print()
            print(node_temp1)
            # check if the two results are the same
            print(np.allclose(node_temp, node_temp1))
            input("here2")
            # record temperature
            temps[t] = node_temp
        ####################################################
        # get mse loss between data and temps
        mse = np.sum((temp_data - temps) ** 2)
        print("here 2")
        # Get mse of 1st derivative
        d_mse = np.sum((np.diff(temp_data) - np.diff(temps)) ** 2)

        mse = mse + d_mse

        return temps
    
    # @partial(jit, static_argnums=(0,))
    def compute_k_matrix(self,model_params, heated_node):

        x_correction_factor = model_params[4] # Define your x correction factor here
        y_correction_factor = model_params[5]  # Define your y correction factor here

        k0 = model_params[-2]
        sigma = model_params[-1]

        k_s =  model_params[0]
        k_b = model_params[1]

        
        t_h = model_params[2]
        t_e = model_params[3]

        

        

        ################## Calculate_distance_matrix ##################

        # Convert node_tup_list to a numpy array for vectorized operations
        node_array = np.array(self.node_tup_list)
        node_array = node_array[:, :-1] # Remove the layer index


        # Add the correction factors to the x and y coordinates
        corrected_node_array = node_array + np.array([x_correction_factor, y_correction_factor])

        # Calculate the differences between each pair of x and y coordinates
        x_diff = corrected_node_array[:, 0, None] - node_array[:, 0]
        y_diff = corrected_node_array[:, 1, None] - node_array[:, 1]

        # Calculate the distances
        distance_matrix = np.sqrt(x_diff**2 + y_diff**2)
        
        ################## Create_k_matrix ##################
        def gaussian_cone_k(distance,sigma,k0):
            return k0 * np.exp(-distance**2/(2*sigma**2))
        
        internal_nodes = self.internal_nodes
        boundary_nodes = self.boundary_nodes

        k_mat = np.zeros( (len(internal_nodes),len(boundary_nodes)+len(internal_nodes)), dtype='float64')

        for i in range(len(self.edge_indices)):
            # k_val = model_params[i]
            node_index, neighbor_index = self.edge_indices[i]
            if node_index < len(internal_nodes):
                if neighbor_index == len(internal_nodes): # node is connected to heater node
                    if heated_node == -1:
                        k_val = 0
                    else:
                        k_val = gaussian_cone_k(distance_matrix[heated_node][node_index], sigma, k0)
                    k_mat[node_index, neighbor_index] = k_val
                elif neighbor_index < len(internal_nodes): # node is connected to another internal node
                    k_mat[node_index, neighbor_index] = model_params[0]
                else: # node is connected to boundary node
                    k_mat[node_index, neighbor_index] = model_params[1]
                # k_mat = k_mat.at[node_index, neighbor_index].set(k_val)
            if neighbor_index < len(internal_nodes):
                # k_mat = k_mat.at[neighbor_index, node_index].set(k_val)
                if(node_index == len(internal_nodes)):
                    if heated_node == -1:
                        k_val = 0
                    else:
                        k_val = gaussian_cone_k(distance_matrix[heated_node][neighbor_index], sigma, k0)
                    k_mat[neighbor_index, node_index] = k_val
                elif neighbor_index < len(internal_nodes): # node is connected to another internal node
                    k_mat[neighbor_index, node_index] = model_params[0]
                else: # node is connected to boundary node
                    k_mat[neighbor_index, node_index] = model_params[1]
            
            # add k val to diagonal
            if node_index < len(internal_nodes):
                k_mat[node_index, node_index] = k_mat[node_index, node_index] - model_params[0]
            if neighbor_index < len(internal_nodes):
                k_mat[neighbor_index, neighbor_index] = k_mat[neighbor_index, neighbor_index] - model_params[0]

        return k_mat



    # @jit
    # def gaussian_cone_k(distance, sigma, k0):
    #     return k0 * np.exp(-distance**2/(2*sigma**2))

    # @partial(jit, static_argnums=(0,))
    # def compute_k_matrix1(self, model_params, heated_node):
    #     def gaussian_cone_k(distance, sigma, k0):
    #         return k0 * np.exp(-distance**2/(2*sigma**2))
    #     x_correction_factor = model_params[4]
    #     y_correction_factor = model_params[5]
    #     k0 = model_params[-2]
    #     sigma = model_params[-1]
    #     k_s = model_params[0]
    #     k_b = model_params[1]

    #     node_array = np.array(self.node_tup_list)[:, :-1]
    #     corrected_node_array = node_array + np.array([x_correction_factor, y_correction_factor])
    #     x_diff = corrected_node_array[:, 0, None] - node_array[:, 0]
    #     y_diff = corrected_node_array[:, 1, None] - node_array[:, 1]
    #     distance_matrix = np.sqrt(x_diff**2 + y_diff**2)

    #     internal_nodes = self.internal_nodes
    #     boundary_nodes = self.boundary_nodes
    #     k_mat = np.zeros((len(internal_nodes), len(boundary_nodes) + len(internal_nodes)), dtype='float64')

    #     def update_k_mat(i,k_mat):
    #         node_index, neighbor_index = self.edge_indices[i]
    #         k_val = gaussian_cone_k(distance_matrix[heated_node][node_index], sigma, k0) if neighbor_index == len(internal_nodes) else model_params[0 if neighbor_index < len(internal_nodes) else 1]
    #         k_mat = k_mat.at[node_index, neighbor_index].set(k_val)
    #         k_mat = k_mat.at[neighbor_index, node_index].set(k_val)
    #         if node_index < len(internal_nodes):
    #             k_mat = k_mat.at[node_index, node_index].set(k_mat[node_index, node_index] - model_params[0])
    #         if neighbor_index < len(internal_nodes):
    #             k_mat = k_mat.at[neighbor_index, neighbor_index].set(k_mat[neighbor_index, neighbor_index] - model_params[0])
    #         return k_mat
    #     # length = len(self.edge_indices)
    #     # k_mat = vmap(update_k_mat)(np.arange(length))
    #     k_mat = lax.fori_loop(0, len(self.edge_indices), update_k_mat, np.zeros_like(self.edge_indices))

    #     return k_mat


    def simulate_model_dt(self,current_node_temps,heated_node,model_params,dt,k_mat):
        
        
        # if self.cached_k_matrix.get(heated_node) is None:

        

        node_temp = current_node_temps
        boundary_temp = np.zeros(len(self.boundary_nodes), dtype='float64')
        
        for bound_in in range(len(self.boundary_nodes)):
            node_ind = bound_in + len(self.edges)
            if node_ind == len(self.edges):
                boundary_temp[bound_in] = model_params[2]
            else:
                boundary_temp[bound_in] = model_params[3]
        
        catenat = np.hstack((node_temp, boundary_temp))

        node_temp = node_temp + dt * k_mat @ catenat

        return node_temp

    def simulate_model_horizon(self,initial_conditions,heated_node,model_params,horizon,dt):    
        num_steps = int(horizon/dt)
        temps = np.zeros((num_steps, len(self.internal_nodes)), dtype='float64')
        temps[0] = initial_conditions.flatten()
        k_mat = self.compute_k_matrix(model_params, heated_node)
        for t in range(1,num_steps):
            temps[t] = self.simulate_model_dt(temps[t-1],heated_node,model_params,dt,k_mat)

        return temps

    def evaluate_model_loss(self,model_params,dt,data):
        
        temp_data = data[:, :-1]
        heated_nodes = data[:, -1].reshape(-1).astype(int)
        num_steps = len(data)-1

        temps = np.zeros((len(temp_data), len(self.internal_nodes)), dtype='float64')

        # set first row to initial condition temperatures
        temps[0] = temp_data[0,:]
        k_mat = self.compute_k_matrix(model_params, heated_nodes[0])
        # k_mat1 = self.compute_k_matrix1(model_params, heated_nodes[0])
        # print(k_mat)
        # print()
        # print(k_mat1)
        # print()
        # print(np.allclose(k_mat, k_mat1))
        # input("here")
        prev_heated_node = None
        for t in range(1,num_steps+1):
            current_heated_node = heated_nodes[t]
            if prev_heated_node != current_heated_node:
                k_mat = self.compute_k_matrix(model_params, current_heated_node)
            temps[t] = self.simulate_model_dt(temps[t-1],heated_nodes[t],model_params,dt,k_mat) 
            prev_heated_node = current_heated_node

        

        mse = np.mean((temp_data - temps) ** 2)

        d_mse = np.mean((np.diff(temp_data) - np.diff(temps)) ** 2)

        loss = mse + d_mse

        return loss
    
    def evaluate_model(self,model_params,dt,data):

        data_copy = data.drop(columns=['time'])
        data = data_copy.to_numpy()
       
        temp_data = data[:, :-1]
        heated_nodes = data[:, -1].reshape(-1).astype(int)
        num_steps = len(data)-1

        temps = np.zeros((len(temp_data), len(self.internal_nodes)), dtype='float64')

        # set first row to initial condition temperatures
        temps[0] = temp_data[0,:]
        k_mat = self.compute_k_matrix(model_params, heated_nodes[0])
        # k_mat1 = self.compute_k_matrix1(model_params, heated_nodes[0])
        # print(k_mat)
        # print()
        # print(k_mat1)
        # print()
        # print(np.allclose(k_mat, k_mat1))
        # input("here")
        prev_heated_node = None
        for t in range(1,num_steps+1):
            current_heated_node = heated_nodes[t]
            if prev_heated_node != current_heated_node:
                k_mat = self.compute_k_matrix(model_params, current_heated_node)
            temps[t] = self.simulate_model_dt(temps[t-1],heated_nodes[t],model_params,dt,k_mat)
            prev_heated_node = current_heated_node

        

        return temps

    def learn_model_parameters_new(self,data):
        start = time.time()

        # x_correction_factor = model_params[4] # Define your x correction factor here
        # y_correction_factor = model_params[5]  # Define your y correction factor here

        # k0 = model_params[-2]
        # sigma = model_params[-1]

        # k_s =  model_params[0]
        # k_b = model_params[1]

        
        # t_h = model_params[2]
        # t_e = model_params[3]

        model_params_init = np.array([.1,.1,400,80,.1,.1,.1,1])

        lower_bounds = np.array([0,0,0,0,-1,-1,.0,0.0001])
        upper_bounds = np.array([1,1,800,100,1,1,np.inf,2])
        bounds = list(zip(lower_bounds, upper_bounds))

        data_copy = data.drop(columns=['time'])
        data = data_copy.to_numpy()

        def objective_fn(model_params, data, dt):
            return self.evaluate_model_loss(model_params,dt,data)

        objective_fn_with_args = partial(objective_fn, data=data, dt=0.1)

        result = minimize(self.evaluate_model_loss, model_params_init, args=( .1, data), method='L-BFGS-B',bounds=bounds)

        print(result)
        print("Optimization time:", time.time() - start)
        return result.x

         
    def plot_results(self, data, temp_results):
        data_copy = data.drop(columns=['time'])
        data = data_copy.to_numpy()
        temp_data = data[:, :-1]
        num_steps = len(data)-1
        predicted_results = temp_results
        fit_curve_df = pd.DataFrame()

        fit_curve_df["Time_Seconds"] = np.arange(data.shape[0] - 2) * .1
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        for i in range(temp_results.shape[1]):
            time_steps = np.arange(data.shape[0] - 2) * .1
            axs[0].plot(time_steps,data[:-2, i], label=f'Location {i+1}')
            axs[1].plot(time_steps,predicted_results[:-2, i], label=f'Location {i+1}')
            fit_curve_df["Node_"+str(i)] = predicted_results[:-2, i]
            mae = mean_absolute_error(temp_data, predicted_results)
            mse = mean_squared_error(temp_data, predicted_results)

            # Add MAE and MSE to the plots
            axs[0].text(0.05, 0.95, f'MAE: {mae:.2f}\nMSE: {mse:.2f}', transform=axs[0].transAxes, verticalalignment='top')
            axs[1].text(0.05, 0.95, f'MAE: {mae:.2f}\nMSE: {mse:.2f}', transform=axs[1].transAxes, verticalalignment='top')

        fit_curve_df.to_csv(params["fit_curve_file_path"]+str(row_count)+"_"+str(col_count)+"_"+str(params['heat_node_row'])+"_"+str(params['heat_node_col'])+'_fit_curve.csv', index=False)
        axs[0].set_title('Actual Temperature Results')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Temperature')

        axs[1].set_title('Predicted Temperature Results')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Temperature')

        # Set the same x and y limits for both plots
        ylim = (70, 170)  # Replace with your actual min and max temperatures

        # axs[0].set_xlim(xlim)
        axs[0].set_ylim(ylim)

        # axs[1].set_xlim(xlim)
        axs[1].set_ylim(ylim)

        plt.tight_layout()  # Adjust layout to not overlap subplots
        plt.savefig(curve_fit_plot_file_path+str(row_count)+"x"+str(col_count)+"_"+str(params['heat_node_row'])+"_"+str(params['heat_node_col'])+'_2_subplots.png')  # Save the figure to a file

        plt.figure("matching two plots")
        for i in range(temp_results.shape[1]):
            time_steps = np.arange(data.shape[0] - 2) * .1
            plt.plot(time_steps,data[:-2, i], label=f'Location {i+1}')
            plt.plot(time_steps,predicted_results[:-2, i], label=f'Location {i+1}')

        plt.tight_layout()
        plt.savefig(curve_fit_plot_file_path+str(row_count)+"x"+str(col_count)+"_"+str(params['heat_node_row'])+"_"+str(params['heat_node_col'])+'_pred_fit_curves_match.png')
        plt.show()


        
    def import_training_data(self, data, visualize=True):
        
        # last two columns are x and y index of heated node which needs to be remapped to node index integer from mapping

        
        data.rename(columns={data.columns[0]: 'time'}, inplace=True)
        heated_node = data.apply(lambda row: (int(row[-2]), int(row[-1]),1), axis=1)
        
        
        graph_to_data_mapping_mod = self.graph_to_data_mapping
        graph_to_data_mapping_mod[(-1,-1,1)] = 0
        print(graph_to_data_mapping_mod)
        input("Press Enter to continue...")
        # change heated node to node index
        heated_node = heated_node.apply(lambda row: graph_to_data_mapping_mod[row]-1)
        print(heated_node)
        input("Press Enter to continue...")
        # remove last two columns
        data = data.iloc[:, :-2]


        for col_key in self.data_to_graph_mapping.keys():

            data.rename(columns={data.columns[col_key]: self.data_to_graph_mapping[col_key]}, inplace=True)
        # data.rename(columns={data.columns[-1]: 'heated_node'}, inplace=True)

        # add heated node to the data
        data['heated_node'] = heated_node

        data['time'] = data['time'] - data['time'].iloc[0]
        print(data)
        input("Press Enter to continue...")
        # Create a new DataFrame with the desired time steps
        new_time = np.arange(.1, data['time'].iloc[-1] + 0.1, 0.1)
        new_df = pd.DataFrame({'time': new_time})
        df = pd.concat([data, new_df]).sort_values(by='time').reset_index(drop=True)

        # Interpolate the data
        # df = df.interpolate(method='linear')

        # Interpolate all columns except the last one
        df.iloc[:, :-1] = df.iloc[:, :-1].interpolate(method='linear')

        # Fill the last column with the previous value
        df.iloc[:, -1] = df.iloc[:, -1].fillna(method='ffill')
        
        df['time'] = df['time'].round(decimals=1)
        
        # Remove rows where time is not a multiple of 0.1
        df = df.groupby('time').first().reset_index()

        if visualize:

            plt.figure(figsize=(10, 6))

            for column in df.columns[1:]:  # Skip the first (time) and last (heated_node) columns
                plt.plot(df['time'], df[column], label=column)

            plt.xlabel('Time (seconds)')
            plt.ylabel('Temperature')
            plt.title('Temperature vs Time')
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')  # Move the legend outside the plot

            plt.show()
        

        self.data = df
        
        return df

def import_data():
    df = pd.read_csv(dataPath)
    return df

def save_model_params(model_params,row_count,col_count, file_path):
    # Create a dictionary to hold the data
    param_array = np.array(model_params)
    data = {
        'model_params': param_array.tolist(),
        'row_count': row_count,
        'col_count': col_count
    }

    # Open the file in write mode and dump the data
    with open(file_path, 'w') as file:
        yaml.dump(data, file)

def learn_evaluate():
    file_paths = [dataPath, k_value_data_file_path, curve_fit_plot_file_path]
    ensure_directories_exist(file_paths)
    fea = FEAProblem()
    data = import_data()
    print(data)
    input("Press Enter to continue...")
    therm_model =ThermModel(fea)
    data = therm_model.import_training_data(data)

    # therm_model.learn_model_parameters()
    params = therm_model.learn_model_parameters_new(data)

    save_model_params(params, row_count, col_count, 'model_sheet_params.yaml')

    # params = [ 7.505e-02,  1.672e-01,  4.000e+02,  8.067e+01,  1.907e-01, 6.859e-02,  3.682e-02,  6.261e-01]

    temps = therm_model.evaluate_model(params, .1, data)

    therm_model.plot_results(data, temps)

def import_evaluate():


    # test function importing model_params and using it to evaluate the model and plot the results
    with open('heat_ws/robot_surface_heating/src/heat_planning/scripts/model_sheet_params.yaml', 'r') as file:
        model_sheet_params = yaml.safe_load(file)
    
    # file_paths = [dataPath, k_value_data_file_path, curve_fit_plot_file_path]
    # ensure_directories_exist(file_paths)
    fea = FEAProblem(model_sheet_params=model_sheet_params)
    # data = import_data()

    therm_model =ThermModel(fea, model_sheet_params=model_sheet_params)

    # data = therm_model.import_training_data(data)

    # temps = therm_model.evaluate_model(therm_model.model_parameters, .1, data)
    
    # therm_model.plot_results(data, temps)
   

def main():
    # learn_evaluate()
    import_evaluate()

if __name__ == "__main__":
    main()

