import json
import os
import numpy as np
import cv2
import random
import math
import copy
from agent_memory import AgentMemory

from LLM.LLM import LLM
from channel.tokchan import TokenChannel

CELL_SIZE = 0.125
ANGLE = 15
EXPLORE_ROTATION_STEPS = max(1, int(math.ceil(360.0 / ANGLE)))

class lm_agent:
    def  __init__(self, agent_id, logger, max_frames, args, output_dir = 'results'):
        self.with_oppo = None
        self.oppo_pos = None
        self.with_character = None
        self.color2id = None
        self.satisfied = None
        self.object_list = None
        self.container_held = None
        self.gt_mask = None
        self.object_info = {} # {id: {id: xx, type: 0/1/2, name: sss, position: x,y,z}}
        self.object_per_room = {} # {room_name: {0/1/2: [{id: xx, type: 0/1/2, name: sss, position: x,y,z}]}}
        self.id_map = None
        self.object_map = None
        self.agent_id = agent_id
        self.agent_type = 'lm_agent'
        self.agent_names = ["Alice", "Bob"]
        self.opponent_agent_id = 1 - agent_id
        self.env_api = None
        self.max_frames = max_frames
        self.output_dir = output_dir
        self.map_size = (240, 120)
        self.save_img = True
        self._scene_bounds = {
            "x_min": -15,
            "x_max": 15,
            "z_min": -7.5,
            "z_max": 7.5
        }
        self.max_nav_steps = 80
        self.max_move_steps = 150
        self.logger = logger
        random.seed(1024)
        self.debug = True

        self.new_object_list = None
        self.visible_objects = None
        self.num_frames = None
        self.steps = None
        self.obs = None
        self.local_step = 0

        self.last_action = None
        self.pre_action = None

        self.goal_objects = None
        self.dropping_object = None

        self.source = args.source
        self.lm_id = args.lm_id
        self.prompt_template_path = args.prompt_template_path
        self.communication = args.communication
        self.token_comm = args.token_comm
        self.think = args.think
        self.args = args
        self.LLM = LLM(self.source, self.lm_id, self.think, self.prompt_template_path, self.communication, self.token_comm, self.agent_id)

        self.token_channel = None
        if self.communication:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            default_channel_dir = os.path.join(repo_root, "channel")
            default_snr_to_ter_csv = os.path.join(default_channel_dir, "snr_ber_ter.csv")
            tokenizer_dir = getattr(self.args, "comm_tokenizer_dir", default_channel_dir)
            snr_db = getattr(self.args, "comm_snr_db", 8.0)
            snr_to_ter_csv_path = getattr(self.args, "comm_snr_to_ter_csv_path", default_snr_to_ter_csv)
            self.token_channel = TokenChannel(
                tokenizer_dir=tokenizer_dir,
                snr_db=snr_db,
                snr_to_ter_csv_path=snr_to_ter_csv_path,
                transmission_log_path=output_dir,
                enable_noise=bool(getattr(self.args, "enable_noise", False)),
            )

        self.action_history = []
        self.dialogue_history = []
        self.plan = None
        self.received_message_this_step = False
        self.last_received_message = None
        self.last_received_from = None
        self.last_observed_messages = {}
        self.pending_replanning_context = ""
        self.message_cooldown_steps = 10
        self.message_memory_refresh_interval = max(1, int(getattr(args, "message_memory_refresh_interval", 5)))
        self.startup_delay_steps = max(0, int(getattr(args, "agent1_startup_delay_steps", 1))) if agent_id == 1 else 0
        self.last_message_sent_frame = None
        self.partner_message_memory = None
        self.partner_message_memory_summary = ""
        self.stable_dialogue_memory_due = False

        self.rooms_name = None
        self.rooms_explored = {}
        self.position = None
        self.forward = None
        self.current_room = None
        self.holding_objects_id = None
        self.oppo_holding_objects_id = None
        self.locally_unavailable_object_ids = None
        self.oppo_last_room = None
        self.rotated = None
        self.explore_count = 0
        self.explore_state = None
        self.room_mask_cache = {}
        self.navigation_threshold = 5
        self.detection_threshold = 5
        self.consumed_object_ids = None

    def _estimate_channel_message_stats(self, message: str):
        if self.token_channel is None:
            return self.LLM.estimate_free_text_stats(message)
        return self.token_channel.estimate_text_stats(message)

    def _build_decode_context(self) -> str:
        recent_dialogue = self.LLM.dialogue_history[-6:]
        recent_dialogue_text = "\n".join(recent_dialogue) if recent_dialogue else "None"
        stable_memory = self.partner_message_memory_summary or "No stable dialogue memory yet."
        current_room = self.current_room or "Unknown"
        return (
            f"Goal: {self.LLM.goal_desc or 'Unknown'}\n"
            f"Current room: {current_room}\n"
            f"Stable dialogue memory:\n{stable_memory}\n\n"
            f"Recent decoded dialogue:\n{recent_dialogue_text}"
        )

    def _build_protocol_task_inventory(self):
        if self.env_api is None or 'get_room_ground_truth_object_infos' not in self.env_api:
            return {}

        inventory = {}
        for room_name in self.rooms_name or []:
            if room_name is None:
                continue

            targets = []
            containers = []
            seen_targets = set()
            seen_containers = set()
            try:
                room_objects = self.env_api['get_room_ground_truth_object_infos'](room_name) or []
            except Exception:
                room_objects = []

            for obj in room_objects:
                if not isinstance(obj, dict):
                    continue
                object_id = obj.get("id")
                object_type = obj.get("type")
                object_name = str(obj.get("name", "object")).strip() or "object"
                if object_id is None:
                    continue
                entry = f"<{object_name}> ({object_id})"
                if object_type == 0 and entry not in seen_targets:
                    seen_targets.add(entry)
                    targets.append(entry)
                elif object_type == 1 and entry not in seen_containers:
                    seen_containers.add(entry)
                    containers.append(entry)

            inventory[room_name] = {
                "targets": sorted(targets),
                "containers": sorted(containers),
            }

        return inventory

    def _set_replanning_context(self, context: str) -> None:
        self.pending_replanning_context = str(context).strip()

    def _consume_replanning_context(self) -> str:
        context = self.pending_replanning_context
        self.pending_replanning_context = ""
        return context

    def _handle_message_interruption(self) -> None:
        if not self.received_message_this_step:
            return

        sender = self.last_received_from or self.agent_names[self.opponent_agent_id]
        received_message = self.last_received_message or ""
        if self.plan is not None and not self.plan.startswith('send a message:') and not self.plan.startswith('wait'):
            interrupted_plan = self.plan
            self._mark_last_action_interrupted('message')
            self.plan = None
            self._set_replanning_context(
                f"Your previous action plan was '{interrupted_plan}'. It was interrupted because you just received a message from {sender}: '{received_message}'."
            )
            return

        self._set_replanning_context(
            f"You just received a message from {sender}: '{received_message}'."
        )

    def _initialize_partner_message_memory(self):
        self.partner_message_memory = {
            "rooms": {},
            "partner_taken_objects": [],
        }
        self.partner_message_memory_summary = "No stable dialogue memory yet."
        self.stable_dialogue_memory_due = False

    def _describe_object_brief(self, object_id, fallback_name: str = "object") -> str:
        info = self.object_info.get(object_id, {}) if isinstance(self.object_info, dict) else {}
        object_name = str(info.get("name") or fallback_name).strip() or fallback_name
        return f"<{object_name}> ({object_id})"

    def _get_opponent_held_object_ids(self):
        if self.env_api is None or 'get_opponent_held_object_ids' not in self.env_api:
            return set()
        return set(self.env_api['get_opponent_held_object_ids']() or [])

    def _is_env_consumed_object(self, object_id) -> bool:
        if object_id is None or self.env_api is None or 'is_object_consumed' not in self.env_api:
            return False
        try:
            return bool(self.env_api['is_object_consumed'](object_id))
        except Exception:
            return False

    def _clear_object_ids_from_maps(self, object_ids) -> None:
        if self.id_map is None or self.object_map is None or object_ids is None:
            return
        for object_id in object_ids:
            if object_id is None:
                continue
            mask = np.where(self.id_map == object_id)
            self.object_map[mask] = 0
            self.id_map[mask] = 0

    def _remember_object(self, object_info, mark_as_new: bool = False, inject_visible: bool = False) -> bool:
        if not isinstance(object_info, dict):
            return False

        object_id = object_info.get('id')
        object_type = object_info.get('type')
        position = object_info.get('position')
        name = object_info.get('name')
        if object_id is None or object_type not in (0, 1, 2) or position is None:
            return False

        position = np.array(position, dtype=np.float32)
        was_new = object_id not in self.object_info
        self.object_info[object_id] = {
            'id': object_id,
            'type': object_type,
            'name': name,
            'position': position,
        }

        i, j = self.pos2map(position[0], position[2])
        if self.object_map[i, j] == 0 or self.id_map[i, j] == object_id:
            self.object_map[i, j] = object_type + 1
            self.id_map[i, j] = object_id

        if mark_as_new and was_new:
            self.new_object_list[object_type].append(object_id)
        if inject_visible:
            self._inject_visible_object_proxy(self.object_info[object_id])
        return was_new

    def _inject_visible_object_proxy(self, object_info) -> None:
        if self.obs is None or 'visible_objects' not in self.obs:
            return
        visible_objects = self.obs['visible_objects']
        object_id = object_info['id']
        for existing in visible_objects:
            if existing.get('id') == object_id:
                return
        proxy = {
            'id': object_id,
            'type': object_info['type'],
            'seg_color': None,
            'name': object_info['name'],
        }
        for idx, existing in enumerate(visible_objects):
            if existing.get('id') is None:
                visible_objects[idx] = proxy
                return

    def _supplement_room_memory_after_explore(self, room_name: str) -> int:
        if self.env_api is None or 'get_room_ground_truth_object_infos' not in self.env_api:
            return 0
        discovered = 0
        added_objects = []
        try:
            room_objects = self.env_api['get_room_ground_truth_object_infos'](room_name) or []
        except Exception:
            return 0

        unavailable_object_ids = (
            self.consumed_object_ids
            | set(self.holding_objects_id)
            | set(self.locally_unavailable_object_ids)
            | set(self.with_character)
            | self._get_opponent_held_object_ids()
        )
        for object_info in room_objects:
            object_id = object_info.get('id')
            if object_id in unavailable_object_ids:
                continue
            was_new = self._remember_object(object_info, mark_as_new=True, inject_visible=True)
            if was_new:
                discovered += 1
                added_objects.append(self._describe_object_brief(object_id, object_info.get('name', 'object')))
        if discovered > 0:
            summary = f"Explore supplement in {room_name}: added {', '.join(added_objects)}"
            self.logger.info(summary)
            self.LLM._append_io_log({
                "stage": "explore_supplement",
                "room_name": room_name,
                "num_added_objects": discovered,
                "added_objects": added_objects,
                "frame": self.num_frames,
                "summary": summary,
            })
        return discovered

    def _set_grasp_failed_replanning_context(self, object_id: int) -> None:
        self.locally_unavailable_object_ids.add(object_id)
        self._clear_object_ids_from_maps([object_id])
        object_desc = self._describe_object_brief(object_id)
        self._set_replanning_context(
            f"Grasp failed for {object_desc}. The object is no longer available here and has likely already been taken away by the partner agent."
        )

    def _format_partner_message_memory(self) -> str:
        memory = self.partner_message_memory
        if memory is None:
            return "No stable dialogue memory yet."

        room_memory = memory.get("rooms", {}) if isinstance(memory, dict) else {}
        partner_taken_objects = []
        if isinstance(memory, dict):
            raw_taken_objects = memory.get("partner_taken_objects", [])
            if isinstance(raw_taken_objects, list):
                partner_taken_objects = sorted(set(str(x).strip() for x in raw_taken_objects if str(x).strip()))

        if not room_memory and not partner_taken_objects:
            return "No stable dialogue memory yet."

        facts = []
        if partner_taken_objects:
            facts.append(f"- Partner has taken: {', '.join(partner_taken_objects)}.")
        for room in (self.rooms_name or []):
            info = room_memory.get(room, {})
            state = str(info.get("state", "unexplored")).strip() or "unexplored"
            objectives = sorted(set(info.get("objectives", [])))
            confirmed_by = [str(x).strip() for x in info.get("confirmed_by", []) if str(x).strip()]
            attribution = ""
            if confirmed_by:
                attribution = f", confirmed by {', '.join(confirmed_by)}"
            if objectives:
                facts.append(f"- {room}: objectives {', '.join(objectives)}{attribution}.")
            else:
                facts.append(f"- {room}: {state}{attribution}.")

        if not facts:
            return "No stable dialogue memory yet."
        return "\n".join(facts)

    def _refresh_partner_message_memory_summary_if_needed(self, force: bool = False) -> None:
        if self.partner_message_memory is None:
            self._initialize_partner_message_memory()

        if not self.stable_dialogue_memory_due:
            return

        if not self.LLM.dialogue_history:
            return

        try:
            refreshed = self.LLM.refresh_dialogue_memory(
                existing_memory=self.partner_message_memory,
                dialogue_history=self.LLM.dialogue_history,
                room_names=self.rooms_name,
            )
            self.partner_message_memory = refreshed.get("memory", self.partner_message_memory)
            self.partner_message_memory_summary = self._format_partner_message_memory()
            self.LLM.dialogue_history = []
            self.dialogue_history = []
            self.stable_dialogue_memory_due = False
        except Exception as exc:
            self.logger.warning(f"LLM partner memory refresh failed: {exc}")
            self.partner_message_memory_summary = self._format_partner_message_memory()

    def _mark_stable_dialogue_memory_due_if_needed(self) -> None:
        if len(self.LLM.dialogue_history) >= self.message_memory_refresh_interval:
            self.stable_dialogue_memory_due = True


    def pos2map(self, x, z):
        i = int(round((x - self._scene_bounds["x_min"]) / CELL_SIZE))
        j = int(round((z - self._scene_bounds["z_min"]) / CELL_SIZE))
        return i, j

    def map2pos(self, i, j):
        x = i * CELL_SIZE + self._scene_bounds["x_min"]
        z = j * CELL_SIZE + self._scene_bounds["z_min"]
        return x, z

    def get_pc(self, color):
        depth = self.obs['depth'].copy()
        for i in range(len(self.obs['seg_mask'])):
            for j in range(len(self.obs['seg_mask'][0])):
                if (self.obs['seg_mask'][i][j] != color).any():
                    depth[i][j] = 1e9
        #camera info
        FOV = self.obs['FOV']
        W, H = depth.shape
        cx = W / 2.
        cy = H / 2.
        fx = cx / np.tan(math.radians(FOV / 2.))
        fy = cy / np.tan(math.radians(FOV / 2.))

        #Ego
        x_index = np.linspace(0, W - 1, W)
        y_index = np.linspace(0, H - 1, H)
        xx, yy = np.meshgrid(x_index, y_index)

        xx = (xx - cx) / fx * depth
        yy = (yy - cy) / fy * depth

        index = np.where((depth > 0) & (depth < 10))
        xx = xx[index].copy().reshape(-1)
        yy = yy[index].copy().reshape(-1)
        depth = depth[index].copy().reshape(-1)

        pc = np.stack((xx, yy, depth, np.ones_like(xx)))

        pc = pc.reshape(4, -1)

        E = self.obs['camera_matrix']
        inv_E = np.linalg.inv(np.array(E).reshape((4, 4)))
        rot = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        inv_E = np.dot(inv_E, rot)
        rpc = np.dot(inv_E, pc)
        return rpc[:3]

    def cal_object_position(self, o_dict):
        pc = self.get_pc(o_dict['seg_color'])
        if pc.shape[1] < 5:
            return None
        position = pc.mean(1)
        return position[:3]


    def filtered(self, all_visible_objects):
        visible_obj = []
        for o in all_visible_objects:
            if o['type'] is not None and o['type'] < 4:
                visible_obj.append(o)
        return visible_obj

    def _build_step_info(self):
        return {
            'satisfied': self.satisfied,
            'object_list': self.object_list,
            'new_object_list': self.new_object_list,
            'current_room': self.current_room,
            'visible_objects': self.filtered(self.obs['visible_objects']),
            'obs': {k: v for k, v in self.obs.items() if k not in ['rgb', 'depth', 'seg_mask', 'camera_matrix', 'visible_objects']},
        }

    def _refresh_step_info(self):
        self.get_object_list()
        return self._build_step_info()

    def get_object_list(self):
        self._purge_consumed_object_memory()
        object_list = {0: [], 1: [], 2: []}
        self.object_per_room = {room: {0: [], 1: [], 2: []} for room in self.rooms_name}
        unavailable_object_ids = self.consumed_object_ids | set(self.holding_objects_id) | set(self.locally_unavailable_object_ids) | set(self.with_character) | self._get_opponent_held_object_ids()
        for object_id, obj in self.object_info.items():
            object_type = obj.get('type')
            position = obj.get('position')
            if object_type not in object_list or position is None or object_id in unavailable_object_ids:
                continue
            room = self.env_api['belongs_to_which_room'](position)
            if room is None:
                self.logger.warning(f"obj {obj} not in any room")
                continue
            object_list[object_type].append(obj)
            self.object_per_room[room][object_type].append(obj)
        self.object_list = object_list

    def _is_consumed_candidate(self, object_id):
        return object_id in self.consumed_object_ids or object_id in self.with_character

    def _mark_consumed_objects(self, object_ids):
        self.consumed_object_ids.update(object_ids)
        self._clear_object_ids_from_maps(object_ids)

    def _purge_consumed_object_memory(self) -> None:
        if self.env_api is None or 'is_object_consumed' not in self.env_api:
            return

        consumed_ids = set()
        for object_id in list(self.object_info.keys()):
            try:
                if self.env_api['is_object_consumed'](object_id):
                    consumed_ids.add(object_id)
            except Exception:
                continue

        if consumed_ids:
            self._mark_consumed_objects(consumed_ids)

    def get_new_object_list(self):
        self.visible_objects = self.obs['visible_objects']
        self.new_object_list = {0: [], 1: [], 2: []}
        for o_dict in self.visible_objects:
            if o_dict['id'] is None: continue
            self.color2id[o_dict['seg_color']] = o_dict['id']
            if self._is_env_consumed_object(o_dict['id']):
                self._mark_consumed_objects({o_dict['id']})
                continue
            if self._is_consumed_candidate(o_dict['id']) or o_dict['type'] == 4:
                continue
            position = self.cal_object_position(o_dict)
            if position is None:
                continue
            object_id = o_dict['id']
            self.locally_unavailable_object_ids.discard(object_id)
            if o_dict['type'] == 3: # the agent
                if o_dict['id'] == self.opponent_agent_id:
                    position = self.cal_object_position(o_dict)
                    self.oppo_pos = position
                    if position is not None:
                        oppo_last_room = self.env_api['belongs_to_which_room'](position)
                        if oppo_last_room is not None:
                            self.oppo_last_room = oppo_last_room
                continue
            if self._is_consumed_candidate(object_id):
                continue
            self._remember_object({
                'id': object_id,
                'type': o_dict['type'],
                'name': o_dict['name'],
                'position': position,
            }, mark_as_new=True)

    def color2id_fc(self, color):
        if color not in self.color2id:
            if (color != self.agent_color).any(): 
                return -100 # wall
            else: return self.agent_id # agent
        else: return self.color2id[color]

    def l2_distance(self, st, g):
        return ((st[0] - g[0]) ** 2 + (st[1] - g[1]) ** 2) ** 0.5

    def reach_target_pos(self, target_pos, threshold = 1.0):
        x, _, z = self.obs["agent"][:3]
        gx, _, gz = target_pos
        d = self.l2_distance((x, z), (gx, gz))
        if self.plan.startswith('transport'):
            if self.env_api['belongs_to_which_room'](np.array([x, 0, z])) != self.env_api['belongs_to_which_room'](np.array([gx, 0, gz])):
                return False
            if self.env_api is not None and 'get_2d_distance_to_goal_bounds' in self.env_api:
                dist_to_bounds = self.env_api['get_2d_distance_to_goal_bounds'](np.array([x, 0, z], dtype=np.float32))
                return dist_to_bounds <= threshold + 1e-6
        return d <= threshold + 1e-6

    def _mark_last_action_interrupted(self, reason: str) -> None:
        if len(self.action_history) == 0:
            return
        tag = f" (interrupted by {reason})"
        last = self.action_history[-1]
        if tag in last:
            return
        idx = last.rfind(" at step ")
        if idx != -1:
            last = last[:idx] + tag + last[idx:]
        else:
            last = last + tag
        self.action_history[-1] = last

    def _estimate_path_cost(self, target_pos):
        if target_pos is None:
            return 1e9
        try:
            path, cost = self.agent_memory.find_shortest_path(self.position, target_pos)
            if path is None or len(path) == 0:
                return 1e9
            return float(cost)
        except Exception:
            return 1e9

    def _goal_drop_distance(self):
        return float(self.env_api['get_goal_drop_distance']())

    def _drop_held_object_action(self):
        if self.obs["held_objects"][0]['type'] is not None:
            self.dropping_object += [self.obs["held_objects"][0]['id']]
            if self.obs["held_objects"][0]['type'] == 1:
                self.dropping_object += [x for x in self.obs["held_objects"][0]['contained'] if x is not None]
            return {"type": 5, "arm": "left"}
        self.dropping_object += [self.obs["held_objects"][1]['id']]
        if self.obs["held_objects"][1]['type'] == 1:
            self.dropping_object += [x for x in self.obs["held_objects"][1]['contained'] if x is not None]
        return {"type": 5, "arm": "right"}

    def _sync_satisfied_from_goal_zone(self):
        if self.env_api is None or 'get_delivered_target_ids' not in self.env_api:
            return
        delivered_target_ids = self.env_api['get_delivered_target_ids']()
        if delivered_target_ids is None:
            return
        delivered_container_ids = []
        if 'get_delivered_container_ids' in self.env_api:
            delivered_container_ids = self.env_api['get_delivered_container_ids']()
            if delivered_container_ids is None:
                return
        for object_id in delivered_target_ids:
            if object_id not in self.satisfied:
                self.satisfied.append(object_id)
        self._mark_consumed_objects(set(delivered_target_ids) | set(delivered_container_ids))


    def reset(self, obs, goal_objects = None, output_dir = None, env_api = None, rooms_name = None, agent_color = [-1, -1, -1], agent_id = 0, gt_mask = True, save_img = True):
        self.force_ignore = []
        self.gt_mask = gt_mask
        self.agent_memory = AgentMemory(agent_id = self.agent_id, agent_color = agent_color, output_dir = output_dir, gt_mask=self.gt_mask, gt_behavior=True, env_api=env_api, constraint_type = None, map_size = self.map_size, scene_bounds = self._scene_bounds)
        self.invalid_count = 0
        self.obs = obs
        self.env_api = env_api
        self.agent_color = agent_color
        self.agent_id = agent_id
        self.rooms_name = rooms_name
        self.room_distance = 0
        assert type(goal_objects) == dict
        self.goal_objects = goal_objects
        self._initialize_partner_message_memory()
        self.oppo_pos = None
        if self.communication and output_dir is not None:
            self.output_dir = output_dir
            self.token_channel.set_output_dir(output_dir)
            self.token_channel.set_codebook([])
        self.last_action = None
        self.id_map = np.zeros(self.map_size, np.int32)
        self.object_map = np.zeros(self.map_size, np.int32)

        self.object_info = {}
        self.object_list = {0: [], 1: [], 2: []}
        self.new_object_list = {0: [], 1: [], 2: []}
        self.container_held = None
        self.holding_objects_id = []
        self.oppo_holding_objects_id = []
        self.locally_unavailable_object_ids = set()
        self.with_character = []
        self.with_oppo = []
        self.oppo_last_room = None
        self.satisfied = []
        self.consumed_object_ids = set()
        self.color2id = {}
        self.dropping_object = []
        self.steps = 0
        self.num_frames = 0
        # print(self.obs.keys())
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        self.current_room = self.env_api['belongs_to_which_room'](self.position)
        self.rotated = None
        self.explore_count = 0
        self.explore_state = None
        self.room_mask_cache = {}
        self.rooms_explored = {}
        
        self.plan = None
        self.action_history = [f"go to {self.current_room} at initial step"]
        self.dialogue_history = []
        self.last_received_message = None
        self.last_received_from = None
        self.last_observed_messages = {}
        self.pending_replanning_context = ""
        self.last_message_sent_frame = None
        if self.gt_mask == True:
            self.detection_threshold = 5
        else:
            self.detection_threshold = 3
            from detection import init_detection
            # only here we need to use the detection model, other places we use the gt mask
            # so we put the import here
            self.detection_model = init_detection()
        self.navigation_threshold = 5
        # print(self.rooms_name)
        self.LLM.reset(
            self.rooms_name,
            self.goal_objects,
            protocol_task_inventory=self._build_protocol_task_inventory(),
            output_dir=output_dir,
        )
        self.save_img = save_img

    def move(self, target_pos):
        self.local_step += 1
        action, path_len = self.agent_memory.move_to_pos(target_pos)
        return action

    def _get_room_mask(self, target_room):
        if target_room in self.room_mask_cache:
            return self.room_mask_cache[target_room]
        known_map = self.agent_memory.known_map
        room_mask = np.zeros_like(known_map, dtype=np.uint8)
        for i in range(room_mask.shape[0]):
            for j in range(room_mask.shape[1]):
                x, z = self.agent_memory.map2pos(i, j)
                x += CELL_SIZE * 0.5
                z += CELL_SIZE * 0.5
                pos = np.array([x, 0, z], dtype=np.float32)
                if self.env_api['check_pos_in_room'](pos) and self.env_api['belongs_to_which_room'](pos) == target_room:
                    room_mask[i, j] = 1
        self.room_mask_cache[target_room] = room_mask
        return room_mask

    def _get_unexplored_patch_target(self, target_room, min_ratio = 0.05, min_cells = 16):
        room_mask = self._get_room_mask(target_room) > 0
        navigable = room_mask & (self.agent_memory.wall_map == 0) & (self.agent_memory.true_wall_map == 0) & (self.agent_memory.local_occupancy_map == 0)
        total_area = int(np.count_nonzero(navigable))
        if total_area == 0:
            return None
        unexplored = (self.agent_memory.known_map == 0) & navigable
        if not np.any(unexplored):
            return None
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(unexplored.astype(np.uint8), 4)
        area_threshold = max(min_cells, int(math.ceil(total_area * min_ratio)))
        best_label = None
        best_area = 0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= area_threshold and area > best_area:
                best_label = label
                best_area = area
        if best_label is None:
            return None
        coords = np.argwhere(labels == best_label)
        centroid = centroids[best_label]
        target_ij = coords[np.argmin((coords[:, 0] - centroid[1]) ** 2 + (coords[:, 1] - centroid[0]) ** 2)]
        x, z = self.agent_memory.map2pos(int(target_ij[0]), int(target_ij[1]))
        x += CELL_SIZE * 0.5
        z += CELL_SIZE * 0.5
        return np.array([x, 0, z], dtype=np.float32)

    def gotoroom(self):
        target_room = ' '.join(self.plan.split(' ')[2: 4])
        if target_room[-1] == ',': target_room = target_room[:-1]
        if self.debug:
            print(target_room)
        target_pos = self.env_api['center_of_room'](target_room)
        if self.current_room == target_room and self.room_distance == 0:
            self.plan = None
            return None
        if len(self.new_object_list[0]) + len(self.new_object_list[1]) + len(self.new_object_list[2]) > 0:
            self._mark_last_action_interrupted('new objectives')
            self.new_object_list = {0: [], 1: [], 2: []}
            self.plan = None
            return None
        return self.move(target_pos)


    def goexplore(self):
        target_room = ' '.join(self.plan.split(' ')[-2:])
        if self.explore_state is None or self.explore_state['room'] != target_room:
            self.explore_state = {
                'room': target_room,
                'target_pos': self.env_api['center_of_room'](target_room),
                'center_done': False,
                'patch_count': 0,
            }
            self.rotated = None
            self.explore_count = 0
        if self.explore_state['center_done']:
            next_target = None
            if self.explore_state['patch_count'] < 3:
                next_target = self._get_unexplored_patch_target(target_room)
            if next_target is None:
                self._supplement_room_memory_after_explore(target_room)
                self.rooms_explored[target_room] = 'all'
                self.plan = None
                self.explore_state = None
                self.rotated = None
                return None
            self.explore_state['target_pos'] = next_target
        target_pos = self.explore_state['target_pos']
        self.explore_count += 1
        dis_threshold = 0.5 + self.explore_count / 50
        if not self.reach_target_pos(target_pos, dis_threshold):
            return self.move(target_pos)
        if self.rotated is None:
            self.rotated = 0
        if self.rotated < EXPLORE_ROTATION_STEPS:
            self.rotated += 1
            return {"type": 1}
        self.rotated = None
        if not self.explore_state['center_done']:
            self.explore_state['center_done'] = True
        next_target = None
        if self.explore_state['patch_count'] < 3:
            next_target = self._get_unexplored_patch_target(target_room)
        if next_target is None:
            self._supplement_room_memory_after_explore(target_room)
            self.rooms_explored[target_room] = 'all'
            self.plan = None
            self.explore_state = None
            return None
        self.explore_state['target_pos'] = next_target
        self.explore_state['patch_count'] += 1
        self.explore_count = 0
        return self.move(next_target)

    def gograsp(self):
        target_object_id = int(self.plan.split(' ')[-1][1:-1])
        if target_object_id in self.holding_objects_id:
            self.logger.info(f"successful holding!")
            self._clear_object_ids_from_maps([target_object_id])
            self.plan = None
            return None
        
        if self.target_pos is None:
            self.target_pos = copy.deepcopy(self.object_info[target_object_id]['position'])
        target_object_pos = self.target_pos

        if target_object_id not in self.object_info or target_object_id in self.with_oppo or target_object_id in self.locally_unavailable_object_ids:
            if self.debug:
                self.logger.debug(f"grasp failed. object is not here any more!")
            self._set_grasp_failed_replanning_context(target_object_id)
            self.plan = None
            return None
        if not self.reach_target_pos(target_object_pos):
            return self.move(target_object_pos)
        action = {"type": 3, "object": target_object_id, "arm": 'left' if self.obs["held_objects"][0]['id'] is None else 'right'}
        return action
    
    def goput(self):
        if len(self.holding_objects_id) == 0:
            self.plan = None
            self.with_character = [self.agent_id]
            return None

        current_pos = np.array(self.obs["agent"][:3], dtype=np.float32)
        edge_distance = float(self.env_api['get_2d_distance_to_goal_bounds'](current_pos))
        if edge_distance <= self._goal_drop_distance():
            return self._drop_held_object_action()

        target_pos = self.env_api['get_transport_target_pos'](self.position)
        return self.move(target_pos)

    def putin(self):
        if len(self.holding_objects_id) == 1:
            self.logger.info("Successful putin")
            self.plan = None
            return None
        action = {"type": 4}
        return action
    
    def detect(self):
        detect_result = self.detection_model(self.obs['rgb'][..., [2, 1, 0]])['predictions'][0]
        obj_infos = []
        curr_seg_mask = np.zeros((self.obs['rgb'].shape[0], self.obs['rgb'].shape[1], 3)).astype(np.int32)
        curr_seg_mask.fill(-1)
        for i in range(len(detect_result['labels'])):
            if detect_result['scores'][i] < 0.3: continue
            mask = detect_result['masks'][:,:,i]
            label = detect_result['labels'][i]
            curr_info = self.env_api['get_id_from_mask'](mask = mask, name = self.detection_model.cls_to_name_map(label)).copy()
            if curr_info['id'] is not None:
                obj_infos.append(curr_info)
                curr_seg_mask[np.where(mask)] = curr_info['seg_color']
        curr_with_seg, curr_seg_flag = self.env_api['get_with_character_mask'](character_object_ids = self.with_character)
        curr_seg_mask = curr_seg_mask * (~ np.expand_dims(curr_seg_flag, axis = -1)) + curr_with_seg * np.expand_dims(curr_seg_flag, axis = -1)
        return obj_infos, curr_seg_mask

    def LLM_plan(self):
        if self.communication:
            self._refresh_partner_message_memory_summary_if_needed(force=False)
        replanning_context = self._consume_replanning_context()
        message_cooldown_active = (
            self.last_message_sent_frame is not None
            and (self.num_frames - self.last_message_sent_frame) < self.message_cooldown_steps
        )
        return self.LLM.run(
            self.num_frames,
            self.current_room,
            self.rooms_explored,
            self.obs['held_objects'],
            [self.object_info[x] for x in self.satisfied if x in self.object_info],
            self.object_list,
            self.object_per_room,
            self.action_history,
            self.obs['oppo_held_objects'],
            self.oppo_last_room,
            replanning_context=replanning_context,
            allow_message_action=(
                not self.received_message_this_step
                and len(replanning_context.strip()) == 0
                and not message_cooldown_active
            ),
            partner_memory_desc=self.partner_message_memory_summary,
            goal_position_known=(self.env_api is not None and 'get_goal_position' in self.env_api),
        )

    def get_communication_stats(self):
        if not self.communication or self.token_channel is None:
            return {
                "agent_id": self.agent_id,
                "communication_enabled": bool(self.communication),
                "token_comm_enabled": bool(getattr(self.args, "token_comm", False)),
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
                "observed_ter": 0.0,
            }

        stats = self.token_channel.get_episode_stats()
        stats.update({
            "agent_id": self.agent_id,
            "communication_enabled": bool(self.communication),
            "token_comm_enabled": bool(getattr(self.args, "token_comm", False)),
        })
        return stats

    def act(self, obs):
        self.obs = obs.copy()
        self.obs['rgb'] = self.obs['rgb'].transpose(1, 2, 0)
        self.num_frames = obs['current_frames']
        self.steps += 1

        if not self.gt_mask:
            self.obs['visible_objects'], self.obs['seg_mask'] = self.detect()

        if obs['valid'] == False:
            if self.last_action is not None and 'object' in self.last_action:
                self._clear_object_ids_from_maps([self.last_action['object']])
                if self.last_action.get('type') == 3:
                    self._set_grasp_failed_replanning_context(self.last_action['object'])
            self.invalid_count += 1
            self.plan = None
            assert self.invalid_count < 10, "invalid action for 10 times"
    
        self.received_message_this_step = False
        self.last_received_message = None
        self.last_received_from = None
        if self.communication:
            for i in range(len(obs["messages"])):
                raw = obs["messages"][i]
                if raw is None:
                    continue

                raw_text = str(copy.deepcopy(raw)).strip()
                if self.last_observed_messages.get(i) == raw_text:
                    continue
                self.last_observed_messages[i] = raw_text

                if i == self.agent_id:
                    self.dialogue_history.append(f"{self.agent_names[i]}: {raw_text}")
                    pass
                else:
                    observed_text = raw_text
                    protocol_stats = None
                    mode = "plain_text"
                    receive_success = True
                    observed_tokens = []
                    all_tokens_masked = False
                    if self.token_channel is not None:
                        observed_tokens = self.token_channel.text_to_token_ids(observed_text)
                        all_tokens_masked = self.token_channel.all_tokens_masked(observed_tokens)
                    if getattr(self.args, 'token_comm', False):
                        mode = "protocol"
                        if all_tokens_masked:
                            receive_success = False
                            decoded_message = ""
                            protocol_stats = self.LLM.estimate_protocol_stats(
                                observed_text,
                                allow_mask_token=True,
                            )
                        else:
                            decoded_result = self.LLM.decode_message(
                                observed_text,
                                context=self._build_decode_context(),
                                noise_channel=bool(getattr(self.args, "enable_noise", False)),
                            )
                            decoded_message = decoded_result["message"]
                            protocol_stats = decoded_result.get("protocol_stats")
                    else:
                        decoded_message = observed_text

                    if self.token_channel is not None:
                        semantic_stats = self._estimate_channel_message_stats(decoded_message)
                        self.token_channel.record_transmission(
                            channel_input_text=observed_text,
                            channel_output_text=decoded_message,
                            tx_token_ids=observed_tokens,
                            rx_token_ids=observed_tokens,
                            snr_db=getattr(self.args, "comm_snr_db", None),
                            mode=mode,
                            semantic_stats=semantic_stats,
                            protocol_stats=protocol_stats,
                            agent_id=self.agent_id,
                            step=self.num_frames,
                            direction="receive",
                            peer_agent_id=i,
                            include_in_stats=False,
                        )

                    if not receive_success:
                        self.LLM._append_io_log({
                            "stage": "decode_skipped_all_masked",
                            "sender_agent_id": i,
                            "received_text": observed_text,
                            "frame": self.num_frames,
                        })
                        continue

                    self.received_message_this_step = True
                    self.dialogue_history.append(f"{self.agent_names[i]}: {decoded_message}")
                    self.last_received_from = self.agent_names[i]
                    self.last_received_message = decoded_message
                    self.LLM.dialogue_history.append(f"{self.agent_names[i]}: {decoded_message}")
                    self._mark_stable_dialogue_memory_due_if_needed()

        if self.steps <= self.startup_delay_steps:
            self.last_action = {'type': 'ongoing'}
            return {'type': 'ongoing'}
    
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        current_room = self.env_api['belongs_to_which_room'](self.position)
        if current_room is not None:
            self.current_room = current_room
        self.room_distance = self.env_api['get_room_distance'](self.position)
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            self.rooms_explored[self.current_room] = 'part'
        if self.agent_id not in self.with_character: self.with_character.append(self.agent_id) # DWH: buggy env, need to solve later.
        self.holding_objects_id = []
        self.with_oppo = []
        self.oppo_holding_objects_id = []
        for x in self.obs['held_objects']:
            if x['type'] == 0:
                self.holding_objects_id.append(x['id'])
                if x['id'] not in self.with_character: self.with_character.append(x['id']) # DWH: buggy env, need to solve later.
                # self.with_character.append(x['id'])
            elif x['type'] == 1:
                self.holding_objects_id.append(x['id'])
                if x['id'] not in self.with_character: self.with_character.append(x['id']) # DWH: buggy env, need to solve later.
                #self.with_character.append(x['id'])
                for y in x['contained']:
                    if y is None:
                        break
                    if y not in self.with_character: self.with_character.append(y)
                    #self.with_character.append(y)
        self.with_oppo = list(self._get_opponent_held_object_ids())
        self.oppo_holding_objects_id = list(self.with_oppo)
        for obj in self.with_oppo:
            self.locally_unavailable_object_ids.add(obj)
        self._clear_object_ids_from_maps(self.with_oppo)
        self._clear_object_ids_from_maps([obj for obj in self.with_character if obj != self.agent_id])
        if len(self.dropping_object) > 0 and self.obs['status'] == 1:
            self.logger.info(f"Drop object: {self.dropping_object}")
            self.dropping_object = []
            if len(self.holding_objects_id) == 0:
                self.logger.info("successful drop!")
                self.plan = None
        self._sync_satisfied_from_goal_zone()

        ignore_obstacles = []
        ignore_ids = []
        self.with_character = [self.agent_id]
        temp_with_oppo = []
        for x in self.obs["held_objects"]:
            if x is None or x["id"] is None:
                continue
            self.with_character.append(x["id"])
            if "contained" in x:
                for y in x["contained"]:
                    if y is not None:
                        self.with_character.append(y)

        for x in self.force_ignore:
            self.with_character.append(x)

        temp_with_oppo = list(self.with_oppo)

        ignore_obstacles = self.with_character + ignore_obstacles
        ignore_ids = self.with_character + ignore_ids
        ignore_ids = temp_with_oppo + ignore_ids
        ignore_ids += list(self.consumed_object_ids)
        ignore_obstacles += list(self.consumed_object_ids)

        self.agent_memory.update(
            obs, ignore_ids=ignore_ids, ignore_obstacles=ignore_obstacles, save_img = self.save_img
        )

        self._handle_message_interruption()

        if self.obs['status'] == 0: # ongoing
            return {'type': 'ongoing'}

        self.get_new_object_list()
        print(self.new_object_list)
        info = self._refresh_step_info()

        action = None
        lm_times = 0
        while action is None:
            if self.plan is None:
                info = self._refresh_step_info()
                self.target_pos = None
                self.explore_state = None
                self.explore_count = 0
                self.rotated = None
                if lm_times > 0:
                    print("retrying LM_plan")
                if lm_times > 3: 
                    print("LM_plan too many times, wait for next step")
                    self.plan = "wait"
                    action = None
                    info.update({"LLM": {"parse_exception": "too_many_retries", "Frames": self.num_frames}})
                    break
                plan, a_info = self.LLM_plan()
                if plan is None: # NO AVAILABLE PLANS! Explore from scratch!
                    print("No more things to do!")
                    plan = f"[wait]"
                self.plan = plan
                self.action_history.append(f"{'send a message' if plan.startswith('send a message:') else plan} at step {self.num_frames}")
                a_info.update({"Frames": self.num_frames})
                info.update({"LLM": a_info})
                lm_times += 1
            if self.plan.startswith('go to'):
                action = self.gotoroom()
            elif self.plan.startswith('explore'):
                action = self.goexplore()
            elif self.plan.startswith('go grasp'):
                action = self.gograsp()
            elif self.plan.startswith('put'):
                action = self.putin()
            elif self.plan.startswith('transport'):
                action = self.goput()
            #    self.with_character = [self.agent_id]
            elif self.plan.startswith('send a message:'):
                semantic_message = ' '.join(self.plan.split(' ')[3:])
                self.LLM.dialogue_history.append(f"{self.agent_names[self.agent_id]}: {semantic_message}")
                self._mark_stable_dialogue_memory_due_if_needed()
                semantic_stats = self._estimate_channel_message_stats(semantic_message)
                
                if getattr(self.args, 'token_comm', False):
                    encode_result = self.LLM.encode_message(semantic_message)
                    message_to_send = encode_result["message"]
                    encoded_stats = encode_result
                    mode = "protocol"
                else:
                    message_to_send = semantic_message
                    encoded_stats = None
                    mode = "plain_text"

                channel_record = None
                if self.token_channel is not None:
                    tx_ids = self.token_channel.text_to_token_ids(message_to_send)
                    rx_ids = self.token_channel.apply_token_errors(
                        tx_ids,
                        snr_db=getattr(self.args, "comm_snr_db", None),
                    )
                    observed_message_to_send = self.token_channel.token_ids_to_text(
                        rx_ids,
                        skip_special_tokens=False,
                    )
                    channel_record = self.token_channel.record_transmission(
                        channel_input_text=message_to_send,
                        channel_output_text=observed_message_to_send,
                        tx_token_ids=tx_ids,
                        rx_token_ids=rx_ids,
                        snr_db=getattr(self.args, "comm_snr_db", None),
                        mode=mode,
                        semantic_stats=semantic_stats,
                        protocol_stats=encoded_stats,
                        agent_id=self.agent_id,
                        step=self.num_frames,
                        direction="send",
                        peer_agent_id=self.opponent_agent_id,
                    )
                    message_to_send = observed_message_to_send

                self.LLM._append_io_log({
                    "stage": "message_send_stats",
                    "semantic_message": semantic_message,
                    "mode": mode,
                    "raw_message_stats": semantic_stats,
                    "encoded_message_stats": encoded_stats,
                    "channel_record": channel_record,
                })
                self.last_message_sent_frame = self.num_frames

                action = {"type": 6,
                          "message": message_to_send}
                self.plan = None
            elif self.plan.startswith('wait'):
                action = None
                break
            else:
                raise ValueError(f"unavailable plan {self.plan}")

        info.update({"action": action,
                     "plan": self.plan})
        if self.debug:
            self.logger.info(self.plan)
            self.logger.debug(info)
        self.last_action = action
        return action
