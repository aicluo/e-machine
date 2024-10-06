import numpy as np
import math


"""
    Generate a sequence of outputs and corresponding states based on provided probability matrices

    Parameters:
    states (list): list of state names as strings, same order as probability matrices
                   (ex: ["A", "B", "C"])
    transition_matrix: see above image
    n (int): length of sequence to output

    Returns:
    two lists, one with the outputs and the other with the corresponding states
"""
def e_machine(states, transition_matrix, n):
    output_states = []
    outputs = []

    # pick a random state to start with
    state_index = np.random.choice(len(states))
    output_states.append(states[state_index])

    for i in range(n):
        # get column of probabilities corresponding to state
        state_prob = transition_matrix[state_index]

        # select transition
        result = np.random.choice(len(state_prob), p=state_prob)

        # figure out output and state corresponding to transition
        output = math.floor(result/len(states))
        state_index = result % len(states)

        output_states.append(states[state_index])
        outputs.append(output)
    
    return output_states, outputs


def probability_rederivation(output_states, outputs):
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

def distribution_generator(num_states, num_outputs):
    alpha = [1] * num_states * num_outputs
    return np.random.dirichlet(alpha, num_states)

def reservoir(h_t, x_t, W, v):
    return np.tanh(W @ h_t + v*x_t)

def weight_initialization(m, n, epsilon):
    # generate random matrix
    W = np.random.uniform(low = -1, high = 1, size=[m,n])

    # find largest magnitude eigenvalue
    eig = max(abs(np.linalg.eigvals(W) + epsilon))
    if eig > 1:
        W = W/eig

    # adjust mean to 0
    W = W - np.mean(W)
    
    return W