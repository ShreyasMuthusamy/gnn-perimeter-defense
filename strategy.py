import random
import networkx as nx
import numpy as np
import torch

from game.lib.utils.sensor_utils import extract_sensor_data, extract_neighbor_sensor_data
from utils.dataset import postprocess_action

model = torch.load('./models/test.ckpt')

def strategy(state):
    """
    Defines the attacker's strategy to move towards the closest flag.
    
    Parameters:
        state (dict): The current state of the game, including positions and parameters.
    """
    current_node = state['curr_pos']
    flag_positions = state['flag_pos']
    flag_weights = state['flag_weight']
    agent_params = state['agent_params']
    
    # Extract positions of attackers and defenders from sensor data
    attacker_positions, defender_positions = extract_sensor_data(
        state, flag_positions, flag_weights, agent_params
    )

    S = np.concat((
        np.array(flag_positions),
        np.array(attacker_positions),
        np.array(defender_positions),
    ))
    node_pred = model.forward(torch.tensor(S, dtype=torch.float))
    node_pred = int(postprocess_action(node_pred)[0])

    try:
        # Determine the next node towards the closest flag
        next_node = agent_params.map.shortest_path_to(
            current_node, node_pred, agent_params.speed
        )
        if next_node is None:
            raise nx.NetworkXNoPath("No path found from the graph util function.")
        state['action'] = next_node
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        # Handle cases where the path cannot be found
        print(f"No path found from red agent at node {current_node} to flag at node {node_pred}: {e}")
        neighbor_data = extract_neighbor_sensor_data(state)
        state['action'] = random.choice(neighbor_data)

def map_strategy(agent_config):
    """
    Maps each attacker agent to the defined strategy.
    
    Parameters:
        agent_config (dict): Configuration dictionary for all agents.
        
    Returns:
        dict: A dictionary mapping agent names to their strategies.
    """
    strategies = {}
    for name in agent_config.keys():
        strategies[name] = strategy
    return strategies