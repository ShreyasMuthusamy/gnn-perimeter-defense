import json
import glob
from typing import Optional

import numpy as np

def get_state(game: dict, step: int) -> np.ndarray:
    state = np.zeros(40)

    num_att = np.array(game['attacker_num'])
    for i in range(10):
        if i < num_att:
            # TODO: Ask why some agents are missing
            try:
                agent_pos = game['agents'][f'attacker_{i}']['positions']
                state[i] = agent_pos[step] if step < len(agent_pos) else -1
                state[i+20] = agent_pos[step+1] if step + 1 < len(agent_pos) else -1
            except:
                state[i] = -1
                state[i+20] = -1
        else:
            state[i] = -1
            state[i+20] = -1

    num_def = np.array(game['defender_num'])
    for i in range(10):
        if i < num_def:
            try:
                agent_pos = game['agents'][f'defender_{i}']['positions']
                state[i+10] = agent_pos[step] if step < len(agent_pos) else -1
                state[i+30] = agent_pos[step+1] if step + 1 < len(agent_pos) else -1
            except:
                state[i+10] = -1
                state[i+30] = -1
        else:
            state[i+10] = -1
            state[i+30] = -1
    
    return state

def to_numpy(replay_data: list) -> np.ndarray:
    games = []
    for game in replay_data:
        total_time = game['total_time']
        flag_positions = np.array(game['flag_positions'])
        flag_positions = np.pad(flag_positions, (0, 5 - len(flag_positions)), constant_values=-1)
        states = []
        if total_time > 1: # TODO: Ask why some games have total_time as 0
            for step in range(total_time-1):
                state = np.concat((flag_positions, get_state(game, step)))
                states.append(state[None])
            games.append(np.concat(states, axis=0))
    
    return np.concat(games, axis=0)

def get_data(graph: str, att_strat: Optional[str] = None, def_strat: Optional[str] = None) -> np.ndarray:
    if att_strat is None:
        att_strat = '*'
    
    if def_strat is None:
        def_strat = '*'

    replay_glob = glob.glob(f'sample_data/{graph}/{att_strat}_vs_{def_strat}.json')
    replays = []
    for replay in replay_glob:
        with open(replay) as f:
            data = json.load(f)
            replays.append(to_numpy(data))

    return np.concat(replays, axis=0)
