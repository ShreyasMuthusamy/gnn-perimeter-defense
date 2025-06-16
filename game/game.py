import traceback
import gymnasium as gym
from typing import Optional, Tuple

from lib.core.core import *
from lib.utils.file_utils import get_directories, export_graph_config
from lib.utils.config_utils import load_configuration, load_config_metadata, create_context_with_sensors
from lib.utils.sensor_utils import create_static_sensors
from lib.utils.game_utils import *
import lib.core.time_logger as TLOG


class LeagueEnv(gym.Env):
    def __init__(
            self,
            config_name: str,
            root_dir: str,
            logger: Optional[TLOG.TimeLogger] = None,
            debug: bool = True,
    ):
        self.time_counter = 0
        self.payoff = 0
        self.total_tags = 0
        self.total_captures = 0

        self.config_name = config_name
        self.root_dir = root_dir
        self.debug = debug
        self.ctx = None

        self.logger = logger
        self.initial_step_log = None

        self.max_time = None
        self.flag_positions = None
        self.flag_weights = None
        self.attacker_count = 0
        self.defender_count = 0

        self.reset()

    def _done(self) -> bool:
        return check_termination(
            self.time_counter,
            self.max_time,
            self.attacker_count,
            self.defender_count
        )
    
    def reset(self):
        dirs = get_directories(self.root_dir)
        config = load_configuration(self.config_name, dirs, self.debug)
        success("Loaded configuration successfully", self.debug)

        G = export_graph_config(config, dirs, self.debug)

        # Create static sensor definitions once
        static_sensors = create_static_sensors()

        # Create a new context with sensors for this run
        self.ctx = create_context_with_sensors(
            config,
            G,
            visualization=False,
            static_sensors=static_sensors,
            debug=self.debug,
        )

        # Initialize agents and assign strategies
        agent_config, agent_params_dict = initialize_agents(self.ctx, config)
        success("Initialized agents successfully", self.debug)

        # Configure visualization and initialize flags
        configure_visualization(self.ctx, agent_config, config)
        initialize_flags(self.ctx, config)

        # Retrieve game parameters
        self.max_time = config.get("game", {}).get("max_time", 1000)
        self.flag_positions = config.get("game", {}).get("flag", {}).get("positions", [])
        self.flag_weights = config.get("game", {}).get("flag", {}).get("weights")
        interaction_config = config.get("game", {}).get("interaction", {})
        payoff_config = config.get("game", {}).get("payoff", {})

        # Initialize the time logger
        metadata = load_config_metadata(config)
        if self.logger is not None:
            current_metadata = self.logger.get_metadata()
            merged_metadata = {**current_metadata, **metadata}
            self.logger.set_metadata(merged_metadata)
        success("Initialized time logger", self.debug)

        success(f"Starting game with max time: {self.max_time}", self.debug)

        # Check initial interactions before any moves
        (init_caps, init_tags, self.attacker_count,
         self.defender_count, init_cap_details, init_tag_details) = check_agent_interaction(
            self.ctx,
            G,
            agent_params_dict,
            self.flag_positions,
            interaction_config,
            self.time_counter
        )
        self.total_captures += init_caps
        self.total_tags += init_tags
        self.payoff += compute_payoff(payoff_config, init_caps, init_tags)

        # Log initial state
        self.initial_step_log = {
            "agents": {agent.name: agent.get_state().get("curr_pos") for agent in self.ctx.agent.create_iter()},
            "flag_positions": self.flag_positions,
            "payoff": self.payoff,
            "captures": init_caps,
            "tags": init_tags,
            "tag_details": init_tag_details,
            "capture_details": init_cap_details,
        }
        if self.logger is not None:
            self.logger.log_data(self.initial_step_log, self.time_counter)



# --- Main Game Runner ---
def run_game(config_name: str, root_dir: str, attacker_strategy, defender_strategy, logger: Optional[TLOG.TimeLogger] = None, visualization: bool = False, debug: bool = False) -> Tuple[float, int, int, int]:
    """
    Runs the multi-agent game using the provided configuration.

    Args:
        config_name: Name of the configuration file or path to the file (relative or absolute)
        root_dir: Root directory of the project
        attacker_strategy: Strategy module for attackers
        defender_strategy: Strategy module for defenders
        logger: Optional TimeLogger instance for logging game events
        visualization: Whether to use visualization
        debug: Whether to print debug messages

    Returns:
        tuple: (final_payoff, time, total_captures, total_tags)
    """
    # Initialize variables outside the try to ensure they exist in finally block
    time_counter = 0
    payoff = 0
    total_tags = 0
    total_captures = 0
    ctx = None
    error_message = None

    try:
        dirs = get_directories(root_dir)
        config = load_configuration(config_name, dirs, debug)
        success("Loaded configuration successfully", debug)

        G = export_graph_config(config, dirs, debug)

        # Create static sensor definitions once
        static_sensors = create_static_sensors()

        # Create a new context with sensors for this run
        ctx = create_context_with_sensors(config, G, visualization=False, static_sensors=static_sensors, debug=debug)

        # Initialize agents and assign strategies
        agent_config, agent_params_dict = initialize_agents(ctx, config)
        assign_strategies(ctx, agent_config, attacker_strategy, defender_strategy)
        success("Assigned strategies successfully", debug)

        # Configure visualization and initialize flags
        configure_visualization(ctx, agent_config, config)
        initialize_flags(ctx, config)

        # Retrieve game parameters
        max_time = config.get("game", {}).get("max_time", 1000)
        flag_positions = config.get("game", {}).get("flag", {}).get("positions", [])
        flag_weights = config.get("game", {}).get("flag", {}).get("weights")
        interaction_config = config.get("game", {}).get("interaction", {})
        payoff_config = config.get("game", {}).get("payoff", {})

        # Initialize the time logger
        metadata = load_config_metadata(config)
        if logger is not None:
            current_metadata = logger.get_metadata()
            merged_metadata = {**current_metadata, **metadata}
            logger.set_metadata(merged_metadata)
        success("Initialized time logger", debug)

        success(f"Starting game with max time: {max_time}", debug)

        # Check initial interactions before any moves
        init_caps, init_tags, attacker_count, defender_count, init_cap_details, init_tag_details = check_agent_interaction(ctx, G, agent_params_dict, flag_positions, interaction_config, time_counter)
        total_captures += init_caps
        total_tags += init_tags
        payoff += compute_payoff(payoff_config, init_caps, init_tags)

        # Log initial state
        initial_step_log = {
            "agents": {agent.name: agent.get_state().get("curr_pos") for agent in ctx.agent.create_iter()},
            "flag_positions": flag_positions,
            "payoff": payoff,
            "captures": init_caps,
            "tags": init_tags,
            "tag_details": init_tag_details,
            "capture_details": init_cap_details,
        }
        if logger is not None:
            logger.log_data(initial_step_log, time_counter)

        # Check if game should terminate after initial state
        if check_termination(time_counter, max_time, attacker_count, defender_count):
            success(f"Game terminated at time {time_counter}", debug)
            return payoff, time_counter, total_captures, total_tags

        # Main game loop
        while not ctx.is_terminated():
            time_counter += 1
            next_actions = {}

            try:
                for agent in ctx.agent.create_iter():
                    state = agent.get_state()
                    state.update({"flag_pos": flag_positions, "flag_weight": flag_weights, "agent_params": agent_params_dict.get(agent.name, {}), "time": time_counter, "payoff": payoff, "name": agent.name, "agent_params_dict": agent_params_dict})
                    if hasattr(agent, "strategy") and agent.strategy is not None:
                        try:
                            agent.strategy(state)
                        except Exception as e:
                            error(f"Error executing strategy for {agent.name}: {e}")
                            traceback.print_exc()
                    else:
                        node = ctx.visual.human_input(agent.name, state)
                        state["action"] = node
                    next_actions[agent.name] = state["action"]
            except Exception as e:
                error(f"An error occurred during agent turn: {e}")
                raise e

            # Update agents with their actions
            for agent in ctx.agent.create_iter():
                state = agent.get_state()
                state["action"] = next_actions.get(agent.name, state.get("action", None))
                agent.set_state()

            # Update visualization display (simulate handles pygame events internally)
            ctx.visual.simulate()

            # Log current agent positions (using "curr_pos" from state)
            agent_positions = {agent.name: agent.get_state().get("curr_pos") for agent in ctx.agent.create_iter()}

            # Check interactions (captures, tags, etc.)
            captures, tags, attacker_count, defender_count, capture_details, tag_details = check_agent_interaction(ctx, G, agent_params_dict, flag_positions, interaction_config, time_counter)
            total_captures += captures
            total_tags += tags
            payoff += compute_payoff(payoff_config, captures, tags)

            step_log = {
                "agents": agent_positions,
                "flag_positions": flag_positions,
                "payoff": payoff,
                "captures": captures,
                "tags": tags,
                "tag_details": tag_details,
                "capture_details": capture_details,
            }
            if logger is not None:
                logger.log_data(step_log, time_counter)

            if check_termination(time_counter, max_time, attacker_count, defender_count):
                success(f"Game terminated at time {time_counter}", debug)
                break

        return payoff, time_counter, total_captures, total_tags

    except KeyboardInterrupt:
        warning("Game interrupted by user.")
        error_message = "Game interrupted by user"
        return payoff, time_counter, total_captures, total_tags
    except Exception as e:
        error_msg = f"An error occurred: {e}"
        error(error_msg)
        error_message = f"{error_msg}\n{traceback.format_exc()}"
        traceback.print_exc()
        return payoff, time_counter, total_captures, total_tags
    finally:
        # Always finalize the logger if it exists
        if logger is not None:
            # Include error information in the finalization if an error occurred
            if error_message:
                logger.finalize(payoff=payoff, time=time_counter, total_captures=total_captures, total_tags=total_tags, error=error_message)
            else:
                logger.finalize(payoff=payoff, time=time_counter, total_captures=total_captures, total_tags=total_tags)
        success("Game completed", debug)
