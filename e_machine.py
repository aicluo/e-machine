import numpy as np
import math

def e_machine(states, transition_matrix, n: int):
    """
        Generate a sequence of hidden states and their corresponding emissions 

        Parameters:
            states (list): list of state names as strings, same order as probability matrices
                        (ex: ["A", "B", "C"])
            transition_matrix: see transition_encoding.jpg
            n (int): length of sequence to output

        Returns:
            two lists, one with the hidden states sequence and the other with the corresponding emissions
    """
    hidden_states = []
    emissions = []

    # pick a random state to start with
    # state_index = np.random.choice(len(states))
    state_index = 1
    hidden_states.append(states[state_index])

    for i in range(n):
        # see transition_encoding.jpg to see how this 3-D matrix is stored in 2-D

        # get column of probabilities corresponding to state
        state_prob_list = transition_matrix[state_index]


        # select transition index - this will be chosen with probability state_prob_list 
        result_index = np.random.choice(len(state_prob_list), p=state_prob_list)

        # if you see the encoding, you will see that the first 2 entries correspond to emission 0, and the next two will correspond to emmission 1 
        # thus, we can use the .floor() function to see if the index corresponds to emission 0 or 1!
        emission = math.floor(result_index / len(states))
        emissions.append(emission)

        #similarly, we use the index to find the state it corresponds to per our transition encoding
        state_index = result_index % 2
        hidden_states.append(states[state_index])

    return hidden_states, emissions
def probability_rederivation(output_states: list, outputs: list):
    """
        Rederive the labelled transition matrices based on output and corresponding states

        Parameters:
            output_states (list): list of states corresponding to outputs
            outputs (list): list of outputs generated from labeled transition matrices

        Returns:
            dictionary whose keys are the transitions and whose values are the transition probabilities
    """
    transitions = {}
    for i, output in enumerate(outputs):
        key = f"({output_states[i+1]}, {output} | {output_states[i]})"
        transitions[key] = transitions.get(key, 0) + 1

    # to store lists of transitions from state n
    states = {}
    for key in transitions.keys():
        state = key[-2]
        states[state] = states.get(state, []) + [[key, transitions[key]]]

    probabilities = {}
    for key in states.keys():
        total = sum(n for _, n in states[key])
        for transition, frequency in states[key]:
            probabilities[f"P{transition}"] = frequency/total
    
    return probabilities

def distribution_generator(num_states: int, num_outputs: int):
    """
    Generate random distribution to be used as labeled transition matrices
    """
    alpha = [1] * num_states * num_outputs
    return np.random.dirichlet(alpha, num_states)

def reservoir(h_t, x_t, W, v):
    """
    Reservoir with equation h_{t+1} = tanh(W*h_t + v*x_t)
    """
    return np.tanh(W @ h_t + v*x_t)


def weight_initialization(n, epsilon=0.1):
    """
    Initializes weight matrices
    Weight matrix of hidden input has eigenvalues between -1 and 1 and mean 0

    Parameters:
        n (int): size of weight matrix
        epsilon (float): marginal quantity to make sure the matrix's eigenvalues are between -1 and 1

    Returns:
        W and v, the weight matrix of the hidden input (size [n, n]) and the weight matrix of the external input
"""
    # generate random matrix
    W = np.random.uniform(low = -1, high = 1, size=[n,n])

    if n > 1:
        # find largest magnitude eigenvalue
        eig = max(abs(np.linalg.eigvals(W)))
        if eig > 1:
            W = W/(eig + epsilon)

        # adjust mean to 0
        W = W - np.mean(W)
    
    # generate v
    v = np.random.rand(n,1)

    return W, v
