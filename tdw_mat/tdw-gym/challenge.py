import argparse
import os
import json
import numpy as np
import cv2

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gym
import time
import logging
import sys

# add this dictionary to python env path:
base_path = os.getcwd()
sys.path.append(base_path)
tdw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tdw_mat"))
if tdw_path not in sys.path:
    sys.path.append(tdw_path)

from h_agent import H_agent
from lm_agent import lm_agent

gym.envs.registration.register(
    id='transport_challenge_MA',
    entry_point='tdw_gym:TDW'
)

class Challenge:
    def __init__(self, logger, port, data_path, output_dir, number_of_agents = 2, max_frames = 2500, launch_build = True, screen_size = 512, data_prefix = 'tdw_mat/dataset/nips_dataset/', gt_mask = True, save_img = True, same_room_init = False):
        self.env = gym.make("transport_challenge_MA", port = port, number_of_agents = number_of_agents, save_dir = output_dir, max_frames = max_frames, launch_build = launch_build, screen_size = screen_size, data_prefix = data_prefix, gt_mask = gt_mask, same_room_init = same_room_init)
        self.gt_mask = gt_mask
        self.logger = logger
        self.logger.debug(port)
        self.logger.info("Environment Created")
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.save_img = save_img
        self.data = json.load(open(os.path.join(data_prefix, data_path), "r"))
        self.logger.info("done")

    def _format_object_entries(self, object_entries):
        if len(object_entries) == 0:
            return "[]"
        return "[" + ", ".join(f"<{name}> ({object_id})" for name, object_id in object_entries) + "]"

    def _log_episode_room_inventory(self, episode, episode_output_dir):
        room_entries = []
        for room_index, room_name in self.env.rooms_name.items():
            if room_name is None:
                continue
            room_entries.append((room_index, room_name))

        room_entries.sort(key=lambda x: x[0])
        room_inventory = {
            room_name: {
                "room_index": room_index,
                "targets": [],
                "containers": [],
            }
            for room_index, room_name in room_entries
        }

        for object_id in self.env.target_object_ids:
            if object_id not in self.env.object_manager.transforms:
                continue
            position = np.array(self.env.object_manager.transforms[object_id].position, dtype=np.float32)
            room_name = self.env.belongs_to_which_room(position)
            if room_name is None or room_name not in room_inventory:
                continue
            object_name = self.env.object_names.get(object_id, "unknown")
            room_inventory[room_name]["targets"].append((object_name, object_id))

        for object_id in self.env.container_ids:
            if object_id not in self.env.object_manager.transforms:
                continue
            position = np.array(self.env.object_manager.transforms[object_id].position, dtype=np.float32)
            room_name = self.env.belongs_to_which_room(position)
            if room_name is None or room_name not in room_inventory:
                continue
            object_name = self.env.object_names.get(object_id, "unknown")
            room_inventory[room_name]["containers"].append((object_name, object_id))

        lines = [f"Episode {episode} room inventory at reset:"]
        for _, room_name in room_entries:
            targets = sorted(room_inventory[room_name]["targets"], key=lambda x: (x[0], x[1]))
            containers = sorted(room_inventory[room_name]["containers"], key=lambda x: (x[0], x[1]))
            lines.append(
                f"  {room_name}: "
                f"targets={self._format_object_entries(targets)}, "
                f"containers={self._format_object_entries(containers)}"
            )
        for line in lines:
            self.logger.info(line)
        inventory_path = os.path.join(episode_output_dir, "room_inventory.txt")
        with open(inventory_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _draw_episode_objective_topdown(self, episode, episode_output_dir):
        valid_regions = [(i, region, self.env.rooms_name.get(i)) for i, region in enumerate(self.env.scene_bounds.regions) if self.env.rooms_name.get(i) is not None]
        if not valid_regions:
            return
        x_min = min(region.x_min for _, region, _ in valid_regions)
        x_max = max(region.x_max for _, region, _ in valid_regions)
        z_min = min(region.z_min for _, region, _ in valid_regions)
        z_max = max(region.z_max for _, region, _ in valid_regions)
        pad = 40
        scale = 40
        width = int(np.ceil((x_max - x_min) * scale)) + pad * 2
        height = int(np.ceil((z_max - z_min) * scale)) + pad * 2
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        def world_to_img(x, z):
            px = int(round((x - x_min) * scale)) + pad
            py = height - (int(round((z - z_min) * scale)) + pad)
            return px, py

        for _, region, room_name in valid_regions:
            x1, y1 = world_to_img(region.x_min, region.z_min)
            x2, y2 = world_to_img(region.x_max, region.z_max)
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            cv2.rectangle(canvas, (left, top), (right, bottom), (210, 210, 210), 2)
            cv2.putText(canvas, room_name, (left + 6, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

        lines = [f"Episode {episode} objective positions:"]
        for idx, object_id in enumerate(sorted(self.env.target_object_ids)):
            if object_id not in self.env.object_manager.transforms:
                continue
            position = np.array(self.env.object_manager.transforms[object_id].position, dtype=np.float32)
            object_name = self.env.object_names.get(object_id, "unknown")
            room_name = self.env.belongs_to_which_room(position)
            px, py = world_to_img(float(position[0]), float(position[2]))
            color = (0, 0, 255)
            cv2.circle(canvas, (px, py), 5, color, -1)
            label = f"{object_name} ({object_id})"
            dx = 8
            dy = -8 if idx % 2 == 0 else 16
            cv2.putText(canvas, label, (px + dx, py + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            lines.append(f"  {label}: room={room_name}, pos=({position[0]:.3f}, {position[2]:.3f})")

        cv2.imwrite(os.path.join(episode_output_dir, "objective_topdown.png"), canvas)
        with open(os.path.join(episode_output_dir, "objective_positions.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        self.logger.info(f"Saved episode {episode} objective top-down view to {os.path.join(episode_output_dir, 'objective_topdown.png')}")

    def _collect_agent_communication_stats(self, agents):
        stats = []
        for agent in agents:
            if hasattr(agent, "get_communication_stats"):
                stats.append(agent.get_communication_stats())
        return stats

    def _aggregate_communication_stats(self, agent_stats):
        totals = {
            "num_messages": 0,
            "plain_text_messages": 0,
            "protocol_messages": 0,
            "raw_total_tokens": 0,
            "raw_total_bits": 0,
            "encoded_total_tokens": 0,
            "encoded_total_bits": 0,
            "channel_total_tokens": 0,
            "channel_total_bits": 0,
            "token_error_count": 0,
        }
        for stats in agent_stats:
            for key in totals:
                totals[key] += int(stats.get(key, 0) or 0)

        num_messages = totals["num_messages"]
        channel_total_tokens = totals["channel_total_tokens"]
        encoded_total_tokens = totals["encoded_total_tokens"]
        encoded_total_bits = totals["encoded_total_bits"]

        totals.update({
            "avg_raw_tokens_per_message": (totals["raw_total_tokens"] / num_messages) if num_messages else 0.0,
            "avg_encoded_tokens_per_message": (totals["encoded_total_tokens"] / num_messages) if num_messages else 0.0,
            "avg_raw_bits_per_message": (totals["raw_total_bits"] / num_messages) if num_messages else 0.0,
            "avg_encoded_bits_per_message": (totals["encoded_total_bits"] / num_messages) if num_messages else 0.0,
            "observed_ter": (totals["token_error_count"] / channel_total_tokens) if channel_total_tokens else 0.0,
            "compression_ratio_tokens": (totals["raw_total_tokens"] / encoded_total_tokens) if encoded_total_tokens else None,
            "compression_ratio_bits": (totals["raw_total_bits"] / encoded_total_bits) if encoded_total_bits else None,
            "agents": agent_stats,
        })
        return totals

    def _resolve_protocol_path(self, protocol_path: str) -> str:
        path = str(protocol_path).strip()
        if not path:
            raise ValueError("protocol path is empty")
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(base_path, path))

    def submit(self, agents, logger, eval_episodes):
        total_finish = 0.0
        total_success = 0.0
        if eval_episodes[0] == -1:
            eval_episodes = range(len(self.data))
        num_eval_episodes = len(eval_episodes)

        start = time.time()
        results = {}
        for i, episode in enumerate(eval_episodes):
            start_time = time.time()
            if os.path.exists(os.path.join(self.output_dir, str(episode), 'result_episode.json')):
                with open(os.path.join(self.output_dir, str(episode), 'result_episode.json'), 'r') as f:
                    result = json.load(f)
                total_finish += result['finish'] / result['total']
                total_success += float(result.get("success_rate", 1.0 if result['finish'] == result['total'] else 0.0))
                results[episode] = result
                continue
            # The episode has been evaluated before

            if not os.path.exists(os.path.join(self.output_dir, str(episode))):
                os.makedirs(os.path.join(self.output_dir, str(episode)))
            self.logger.info('Episode {} ({}/{})'.format(episode, i + 1, num_eval_episodes))
            self.logger.info(f"Resetting Environment ... data is {self.data[episode]}")
            episode_output_dir = os.path.join(self.output_dir, str(episode))
            state, info, env_api = self.env.reset(seed=self.data[episode]['seed'], options=self.data[episode], output_dir = episode_output_dir)
            self._log_episode_room_inventory(episode, episode_output_dir)
            self._draw_episode_objective_topdown(episode, episode_output_dir)
            for id, agent in enumerate(agents):
                if type(env_api) == list:
                    curr_api = env_api[id]
                else: curr_api = env_api
                if info['goal_description'] is not None:
                    if agent.agent_type == 'h_agent':
                        agent.reset(goal_objects = info['goal_description'], output_dir = episode_output_dir, env_api = curr_api, agent_color = info['agent_colors'][id], agent_id = id, gt_mask = self.gt_mask, save_img = self.save_img)
                    elif agent.agent_type == 'lm_agent':
                        agent.reset(obs = state[str(id)], goal_objects = info['goal_description'], output_dir = episode_output_dir, env_api = curr_api, agent_color = info['agent_colors'][id], agent_id = id, rooms_name=info['rooms_name'], gt_mask = self.gt_mask, save_img = self.save_img)
                    else:
                        raise Exception(f"{agent.agent_type} not available")
                else:
                    agent.reset(output_dir = episode_output_dir)
            self.logger.info(f"Environment Reset. Took {time.time() - start_time} secs")
            
            # Prepare protocol if token_comm is used
            lm_agents = [agent for agent in agents if agent.agent_type == 'lm_agent']
            if lm_agents and hasattr(lm_agents[0], 'args') and getattr(lm_agents[0].args, 'token_comm', False):
                if getattr(lm_agents[0].args, 'enable_noise', False):
                    protocol_path = self._resolve_protocol_path(
                        getattr(lm_agents[0].args, 'fixed_protocol_path', '')
                    )
                    self.logger.info(f"Loading fixed communication protocol from {protocol_path} ...")
                    with open(protocol_path, "r", encoding="utf-8") as f:
                        protocol = json.load(f)
                    self.logger.info("Fixed protocol loaded for noisy-channel evaluation.")
                else:
                    self.logger.info("Generating communication protocol via LLM...")
                    protocol = lm_agents[0].LLM.generate_protocol()
                    self.logger.info(f"Protocol generated:\n{protocol}")
                for agent in lm_agents:
                    if hasattr(agent, 'LLM'):
                        agent.LLM.set_protocol(protocol)
                    if getattr(agent, "token_channel", None) is not None:
                        agent.token_channel.set_codebook_from_protocol(agent.LLM.protocol_data)

            local_finish = self.env.check_goal()
            done = False
            step_num = 0
            local_reward = 0.0
            while not done:
                step_num += 1
                actions = {}
                if self.save_img: self.env.save_images(os.path.join(self.output_dir, str(episode), 'Images'))
                for agent_id, agent in enumerate(agents):
                    actions[str(agent_id)] = agent.act(state[str(agent_id)])
                state, reward, done, info = self.env.step(actions)
                local_reward += reward
                local_finish = info.get('goal_status', self.env.check_goal())
                self.logger.info(f"Executing step {step_num} for episode: {episode}, actions: {actions}, finish: {local_finish}, frame: {self.env.num_frames}")
                if done:
                    break
            total_finish += local_finish[0] / local_finish[1]
            success_rate = 1.0 if local_finish[0] == local_finish[1] else 0.0
            total_success += success_rate
            agent_comm_stats = self._collect_agent_communication_stats(agents)
            communication_stats = self._aggregate_communication_stats(agent_comm_stats)
            result = {
                "episode": episode,
                "finish": local_finish[0],
                "total": local_finish[1],
                "transport_rate": (local_finish[0] / local_finish[1]) if local_finish[1] else 0.0,
                "success_rate": success_rate,
                "steps": step_num,
                "frame": self.env.num_frames,
                "communication_stats": communication_stats,
            }
            with open(os.path.join(self.output_dir, str(episode), 'result_episode.json'), 'w') as f:
                json.dump(result, f, indent=2)
            results[episode] = result
        avg_finish = total_finish / num_eval_episodes
        avg_success_rate = total_success / num_eval_episodes

        overall_comm_stats = self._aggregate_communication_stats([
            result.get("communication_stats", {})
            for result in results.values()
        ]) if results else self._aggregate_communication_stats([])
        results = {
            "episode_results": results,
            "avg_finish": avg_finish,
            "avg_transport_rate": avg_finish,
            "avg_success_rate": avg_success_rate,
            "communication_stats": overall_comm_stats,
        }
        with open(os.path.join(self.output_dir, 'eval_result.json'), 'w') as f:
            json.dump(results, f, indent=4)
        self.logger.info(f'eval done, avg transport rate {avg_finish}')
        self.logger.info(f'eval done, avg success rate {avg_success_rate}')
        self.logger.info('time: {}'.format(time.time() - start))
        return avg_finish

    def close(self):
        self.env.close()

def init_logs(output_dir, name = 'simple_example'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, "output.log"), encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def main():
    DEFAULT_NO_LAUNCH_BUILD = False
    DEFAULT_COMMUNICATION = False
    DEFAULT_DEBUG = False
    DEFAULT_TOKEN_COMM = False
    DEFAULT_NO_GT_MASK = False
    DEFAULT_THINK = True
    DEFAULT_NO_SAVE_IMG = True
    DEFAULT_SAME_ROOM_INIT = False
    DEFAULT_ENABLE_NOISE = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--experiment_name", type = str, default = "experiment")
    parser.add_argument("--run_id", type=str, default='run') # paper settings: no_comm comm_wo_protocol comm_w_protocol
    parser.add_argument("--data_path", type=str, default="test_env.json")
    parser.add_argument("--data_prefix", type=str, default="tdw_mat/dataset/dataset_test/")
    parser.add_argument("--port", default=10999, type=int)
    parser.add_argument("--agents", nargs='+', type=str, default=("lm_agent", "lm_agent"))
    parser.add_argument("--eval_episodes", nargs='+', default=[0], type=int, help="which episodes to evaluate on")
    parser.add_argument("--max_frames", default=6000, type=int, help="max frames per episode")
    parser.add_argument("--no_launch_build", type=parse_bool_arg, nargs='?', const=True, default=DEFAULT_NO_LAUNCH_BUILD)
    parser.add_argument("--communication", type=parse_bool_arg, nargs='?', const=True, default=DEFAULT_COMMUNICATION)
    parser.add_argument("--debug", type=parse_bool_arg, nargs='?', const=True, default=DEFAULT_DEBUG)

    parser.add_argument("--comm_tokenizer_dir", type=str, default="channel")
    parser.add_argument("--comm_snr_db", type=float, default=2.0, choices=[0.0, 1.0, 2.0, 3.0, 4.0])
    parser.add_argument("--comm_snr_to_ter_csv_path", type=str, default="snr_ber_ter.csv")
    parser.add_argument("--enable_noise", type=parse_bool_arg, nargs='?', const=True, default=DEFAULT_ENABLE_NOISE, help="whether to apply noisy token channel")
    parser.add_argument("--fixed_protocol_path", type=str, default="", help="fixed protocol used in noisy-channel evaluation")
    parser.add_argument("--token_comm", type=parse_bool_arg, nargs='?', const=True, default=DEFAULT_TOKEN_COMM, help="whether to use encode_message and decode_message")
    parser.add_argument("--no_gt_mask", type=parse_bool_arg, nargs='?', const=True, default=DEFAULT_NO_GT_MASK, help="do not use ground truth mask")

    # LLM parameters
    parser.add_argument('--source', type=str, default='gemma', choices=['gemma'])
    parser.add_argument('--lm_id', type=str, default='31b',
                        help='name for gemma4 model name/path (latest or 31b)')
    parser.add_argument('--prompt_template_path', type=str, default='tdw_mat/LLM/prompts/plan_com.csv',
                        help='path to prompt template file')
    parser.add_argument("--think", type=parse_bool_arg, nargs='?', const=True, default=DEFAULT_THINK, help="think before act")

    parser.add_argument("--screen_size", type=int, default=128)
    parser.add_argument("--no_save_img", type=parse_bool_arg, nargs='?', const=True, default=DEFAULT_NO_SAVE_IMG, help="do not save images")
    parser.add_argument("--same_room_init", type=parse_bool_arg, nargs='?', const=True, default=DEFAULT_SAME_ROOM_INIT, help="spawn multiple agents in the same room at episode start")
    parser.add_argument("--message_memory_refresh_interval", type=int, default=5, help="compress dialogue into partner memory every N dialogue messages")
    parser.add_argument("--agent1_startup_delay_steps", type=int, default=0, help="delay agent 1 by N decision steps")
    
    args = parser.parse_args()
    args.number_of_agents = len(args.agents)
    os.makedirs(args.output_dir, exist_ok = True)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(args.output_dir, exist_ok = True)
    args.output_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(args.output_dir, exist_ok = True)
    logger = init_logs(args.output_dir)

    challenge = Challenge(logger, args.port, args.data_path, args.output_dir, args.number_of_agents, args.max_frames, not args.no_launch_build, screen_size = args.screen_size, data_prefix=args.data_prefix, gt_mask = not args.no_gt_mask, save_img = not args.no_save_img, same_room_init = args.same_room_init)
    agents = []
    for i, agent in enumerate(args.agents):
        if agent == 'h_agent':
            agents.append(H_agent(i, logger, args.max_frames, args.output_dir))
        elif agent == 'lm_agent':
            agents.append(lm_agent(i, logger, args.max_frames, args, args.output_dir))
        else:
            pass
    try:
        challenge.submit(agents, logger, args.eval_episodes)
    finally:
        challenge.close()

if __name__ == "__main__":
    main()
    
