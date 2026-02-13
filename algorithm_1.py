# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 12:43:48 2024

@author: Trabajador
"""

import numpy as np
import gurobipy as gb
import time
import auxiliary_functions as aux

# =============================================================================
def computeAmplificationFactor(output_matrix, input_matrix, max_steps, time_limit_iteration, accuracy = 1e-3):
    """
    Calculate amplification factor and related parameters for a given hypergraph.

    Parameters:
    - output_matrix (np.ndarray): The output incidence matrix of the hypergraph.
    - input_matrix (np.ndarray): The input incidence matrix of the hypergraph.
    - max_steps (int): The maximum number of iterations.
    - time_limit_iteration (float): Time limit for each Gurobi optimization model (each step).
    - accuracy (float, optional): The algorithm stops when the MAF difference between iterations is less than the accuracy.
    Returns:
    - x_t (np.ndarray): Optimal intensity vector for the final iteration.
    - alpha_t (float): Final calculated amplification factor.
    - step (int): Number of iterations completed.
    - alphaDict (dict): Dictionary of alpha values per iteration.
    - total_time (float): Total computation time.
    - results_dict (dict): Detailed results for each iteration.
    """
    
    net_incidence_matrix = output_matrix - input_matrix
    num_nodes, num_arcs = net_incidence_matrix.shape
    x_0 = np.ones(num_arcs)
    
    # Vectorized computation of initial alpha
    initial_alpha_vector = np.sum(output_matrix * x_0, axis=1) / np.maximum(np.sum(input_matrix * x_0, axis=1), 1e-15)
    alpha_0 = np.min(initial_alpha_vector)
    
    # -------------------------------------------------------------------------
    def computeAmplificationFactorFixed(previous_alpha, time_limit):
        m = gb.Model("Amplification_Factor_Model")
        x = m.addVars(num_arcs, lb = 0, ub = 1000, name = "x")
        alpha = m.addVar(name = "alpha")
        
        m.setObjective(alpha, gb.GRB.MAXIMIZE)
        
        # Add constraints
        m.addConstrs(
            (alpha <= gb.quicksum(output_matrix[v, a] * x[a] for a in range(num_arcs)) -
             previous_alpha * gb.quicksum(input_matrix[v, a] * x[a] for a in range(num_arcs))
             for v in range(num_nodes)), 
            name = "constraint_amplification_factor")
        # 
        m.addConstrs(
            (gb.quicksum(input_matrix[v, a] * x[a] for a in range(num_arcs)) >= 1 
             for v in range(num_nodes)
            if sum(input_matrix[v, a] for a in range(num_arcs))),
            name = "constraint_min_input")
        
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = time_limit
        m.Params.MIPGap = 0.00
        
        m.optimize()
        
        if m.status != gb.GRB.OPTIMAL:
            raise ValueError("Optimization did not reach optimality. Check infeasibility report.")
        return np.array([x[a].X for a in range(num_arcs)]), alpha.X, m.NumVars, m.NumConstrs
    # -------------------------------------------------------------------------

    stop = False
    step = 0
    alphaDict = {0: alpha_0}
    previous_alpha = alpha_0
    alpha_t = alpha_0
    alpha_old = 0
    start_time = time.time()
    results_dict = {}

    # counter = 1
    cumulative_time = 0
    while not stop:
        # print(counter, previous_alpha)

        iteration_start_time = time.time()
        
        x_t, alphabar, num_vars, num_constraints = computeAmplificationFactorFixed(previous_alpha, time_limit_iteration)
        
        # Vectorized alpha calculation
        alpha_t = np.min(np.sum(output_matrix * x_t, axis=1) / np.maximum(np.sum(input_matrix * x_t, axis=1), 1e-15))
        
        # counter = counter + 1

        iteration_end_time = time.time()
        it_time = iteration_end_time - iteration_start_time
        cumulative_time = cumulative_time + it_time

        results_dict[step] = {
            "x": x_t,
            "alphabar": alphabar,
            "variables": num_vars,
            "constraints": num_constraints,
            "step": step,
            "alpha": alpha_t,
            "time": it_time
        }
        
        print("step:", step + 1, "alpha:", round(previous_alpha, 3), "it_time:", round(it_time, 3), "total_time:", round(cumulative_time, 3))


        if (np.abs(alphabar) < accuracy or step >= max_steps or np.abs(alpha_old - alpha_t) < accuracy):
            stop = True
            alphaDict[step] = alpha_t
            total_time = time.time() - start_time
            return x_t, alpha_t, step, alphaDict, total_time, results_dict
        else:
            alphaDict[step] = alpha_t
            step += 1
            previous_alpha = alpha_t
            alpha_old = alpha_t
# =============================================================================


# =============================================================================
def recordAmplificationData(input_matrix, output_matrix, nameScenario, time_limit_iteration, name_nodes="", accuracy = 1e-3, max_steps = 1000):
    """
    Compute and log amplification factor information.

    Parameters:
    - input_matrix (np.ndarray): Input incidence matrix.
    - output_matrix (np.ndarray): Output incidence matrix.
    - nameScenario (str): Name for output files.
    - time_limit_iteration (float): Time limit per iteration.
    - name_nodes (list, optional): Names of nodes for detailed output.
    - accuracy (float, optional): precision of the algorithm
    - max_steps (int, optional): Maximal number of steps allowed for the algorithm

    Outputs:
    - Writes results to a file.
    """

    # Check autonomy of the network
    self_sufficient_nodes, self_sufficient_arcs, self_sufficient_general = aux.checkSelfSufficiently(input_matrix, output_matrix)
    
    if not self_sufficient_general:
        print("Error: The hypergraph is not self-sufficient.")
        return

    x, alpha, step, alphaDict, time, dict_a_guardar = computeAmplificationFactor(output_matrix, input_matrix, max_steps, time_limit_iteration, accuracy)
        
    info = []
    info.append("All information:\n")
    info.append(f"{dict_a_guardar}\n")
    info.append(f"Number of steps: {step}\n")
    info.append(f"Total time: {time:.2f} seconds\n")
    info.append(f"Average time per iteration: {time/step:.2f} seconds\n" if step else "Average time per iteration: ---\n")
    info.append(f"Q dimension: {input_matrix.shape}\n")
    info.append(f"Amplification factor: {alpha:.6f}\n")
    info.append(f"Intensities: {x}\n")
    
    info.append("Q arcs:\n")
    row_mapping = {i: i for i in range(input_matrix.shape[0])}
    column_mapping = {j: j for j in range(input_matrix.shape[1])}
    
    if not name_nodes:
        arcs = recordArcs(output_matrix, input_matrix, x, row_mapping, column_mapping)
        filename = nameScenario + '_algorithm_1.txt'
    else:
        arcs = recordArcsNamesNodes(output_matrix, input_matrix, x, row_mapping, column_mapping, name_nodes)
        filename = nameScenario + '_algorithm_1_with_names.txt'

    info.extend(arcs)
    
    filename = "output/" + filename
    
    with open(filename, 'w') as f:
        for line in info:
            f.write(line)
# =============================================================================


# =============================================================================
def recordArcs(output_matrix, input_matrix, x, row_mapping, column_mapping):
    """
    Generate string representations of hypergraph arcs from matrix data.

    Parameters:
    - output_matrix (np.ndarray): Output incidence matrix.
    - input_matrix (np.ndarray): Input incidence matrix.
    - x (np.ndarray): Intensity vector.
    - row_mapping (dict): Mapping for nodes indices.
    - column_mapping (dict): Mapping for arc indices.

    Returns:
    - List[str]: List of arc strings.
    """
    num_nodes, num_arcs = input_matrix.shape
    arcs = []

    for j in range(num_arcs):
        source_set = []
        target_set = []
        
        # Gather source set
        for i in range(num_nodes):
            if input_matrix[i, j] > 0.5:
                coef = int(input_matrix[i, j]) if input_matrix[i, j] > 1 else ''
                source_set.append(f"{coef}s{row_mapping[i] + 1}")
        
        # Gather target set
        for i in range(num_nodes):
            if output_matrix[i, j] > 0.5:
                coef = int(output_matrix[i, j]) if output_matrix[i, j] > 1 else ''
                target_set.append(f"{coef}s{row_mapping[i] + 1}")
        
        # Construct arc string
        source_set_str = ' + '.join(source_set) if source_set else ''
        target_set_str = ' + '.join(target_set) if target_set else ''
        arc = f"{source_set_str} -> {target_set_str} {x[column_mapping[j]]}"
        arcs.append(arc + '\n')
    
    return arcs
# =============================================================================


# =============================================================================
def recordArcsNamesNodes(output_matrix, input_matrix, x, row_mapping, column_mapping, name_nodes):
    """
    Generate string representations of hypergraph using nodes names.

    Parameters:
    - output_matrix (np.ndarray): Output incidence matrix.
    - input_matrix (np.ndarray): Input incidence matrix.
    - x (np.ndarray): Intensity vector.
    - row_mapping (dict): Mapping for nodes indices.
    - column_mapping (dict): Mapping for arc indices.
    - name_nodes (list): List of nodes names.

    Returns:
    - List[str]: List of arc strings with nodes names.
    """
    num_nodes, num_arcs = input_matrix.shape
    arcs = []

    for j in range(num_arcs):
        source_set = []
        target_set = []
        
        # Gather source set
        for i in range(num_nodes):
            if input_matrix[i, j] > 0.5:
                coef = int(input_matrix[i, j]) if input_matrix[i, j] > 1 else ''
                source_set.append(f"{coef}{name_nodes[row_mapping[i]]}")
        
        # Gather target set
        for i in range(num_nodes):
            if output_matrix[i, j] > 0.5:
                coef = int(output_matrix[i, j]) if output_matrix[i, j] > 1 else ''
                target_set.append(f"{coef}{name_nodes[row_mapping[i]]}")
        
        # Construct arc string
        source_set_str = ' + '.join(source_set) if source_set else ''
        target_set_str = ' + '.join(target_set) if target_set else ''
        arc = f"{source_set_str} -> {target_set_str} {x[column_mapping[j]]}"
        arcs.append(arc + '\n')
    
    return arcs
# =============================================================================









