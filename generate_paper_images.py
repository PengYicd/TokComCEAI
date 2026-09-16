import os
import json
import time
import cv2
import numpy as np
import shutil

import sys
base_path = os.getcwd()
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "tdw_mat"))
sys.path.append(os.path.join(base_path, "tdw_mat", "tdw-gym"))

import gym
gym.envs.registration.register(
    id='transport_challenge_MA',
    entry_point='tdw_gym:TDW'
)
from tdw.controller import Controller
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.librarian import ModelLibrarian

def capture_topdown_room(env, output_path):
    c = env.controller
    c.communicate([{"$type": "set_screen_size", "width": 2048, "height": 2048},
                   {"$type": "set_floorplan_roof", "show": False}])
    
    regions = env.scene_bounds.regions
    valid_regions = [r for i, r in enumerate(regions) if env.rooms_name.get(i) is not None]
    if not valid_regions:
        valid_regions = regions
        
    x_min = min(region.x_min for region in valid_regions)
    x_max = max(region.x_max for region in valid_regions)
    z_min = min(region.z_min for region in valid_regions)
    z_max = max(region.z_max for region in valid_regions)
    
    center_x = (x_max + x_min) / 2
    center_z = (z_max + z_min) / 2
    
    max_dim = max(x_max - x_min, z_max - z_min)
    cam_height = (max_dim / 1.15) + 5
    
    cam = ThirdPersonCamera(position={"x": center_x, "y": cam_height, "z": center_z},
                            look_at={"x": center_x, "y": 0, "z": center_z},
                            avatar_id="topdown_cam")
    
    tmp_dir = os.path.join(os.path.dirname(output_path), "tmp_room")
    capture = ImageCapture(avatar_ids=["topdown_cam"], path=tmp_dir, pass_masks=["_img"], png=True)
    c.add_ons.extend([cam, capture])
    
    # Give it a few frames to render
    for _ in range(5):
        c.communicate({"$type": "do_nothing"})
    
    # Find the captured image
    cam_dir = os.path.join(tmp_dir, "topdown_cam")
    if os.path.exists(cam_dir):
        files = [f for f in os.listdir(cam_dir) if f.startswith("img_")]
        if files:
            src = os.path.join(cam_dir, sorted(files)[-1])
            shutil.copy(src, output_path)
            
    # cleanup
    c.add_ons.remove(cam)
    c.add_ons.remove(capture)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

def main():
    output_dir = "paper_images"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rooms"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "objects"), exist_ok=True)

    print("Generating object images...")
    name_map = json.load(open("tdw_mat/dataset/name_map.json"))
    
    # Let's use the gym environment directly for rendering objects to reuse the same built process and port
    # instead of creating a raw Controller which might clash or try to download TDW
    env = gym.make("transport_challenge_MA", port=1071, launch_build=True, data_prefix="tdw_mat/dataset/dataset_train/")
    # Load an empty scene
    env.reset(options={"scene": "1a", "layout": "0_0", "task": "food", "seed": 0})
    c = env.controller
    c.communicate([{"$type": "set_screen_size", "width": 1024, "height": 1024}])
    
    cam = ThirdPersonCamera(position={"x": 0, "y": 1.5, "z": 0}, look_at={"x": 0, "y": 0, "z": 0}, avatar_id="a")
    capture = ImageCapture(avatar_ids=["a"], path=os.path.join(output_dir, "objects_raw"), pass_masks=["_img", "_id"], png=True)
    c.add_ons.extend([cam, capture])

    lib = ModelLibrarian("models_core.json")

    # Load an empty scene ONCE
    c.communicate([{"$type": "load_scene", "scene_name": "ProcGenScene"},
                   {"$type": "set_ambient_light_color", "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0}},
                   {"$type": "set_ambient_light_intensity", "intensity": 1.0}])

    for tdw_name, friendly_name in name_map.items():
        print(f"Rendering {friendly_name} ({tdw_name})...")
        record = lib.get_record(tdw_name)
        if not record:
            print(f"Warning: Model {tdw_name} not found.")
            continue
            
        bounds = record.bounds
        max_dim = max(bounds['back']['z'] - bounds['front']['z'], bounds['right']['x'] - bounds['left']['x'])
        cam_height = max_dim * 1.5 + 0.5
        
        cam.teleport({"x": 0, "y": cam_height, "z": 0})
        cam.look_at({"x": 0, "y": 0, "z": 0})
        
        obj_id = c.get_unique_id()
        c.communicate(c.get_add_object(model_name=tdw_name, object_id=obj_id, position={"x": 0, "y": 0, "z": 0}))
        
        c.communicate([{"$type": "send_segmentation_colors", "frequency": "once"}])
        for _ in range(5):
            c.communicate({"$type": "do_nothing"})
        
        raw_dir = os.path.join(output_dir, "objects_raw", "a")
        if os.path.exists(raw_dir):
            img_files = sorted([f for f in os.listdir(raw_dir) if f.startswith("img_")])
            id_files = sorted([f for f in os.listdir(raw_dir) if f.startswith("id_")])
            
            if img_files and id_files:
                img_path = os.path.join(raw_dir, img_files[-1])
                id_path = os.path.join(raw_dir, id_files[-1])
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                id_img = cv2.imread(id_path, cv2.IMREAD_COLOR)
                
                gray_id = cv2.cvtColor(id_img, cv2.COLOR_BGR2GRAY)
                _, alpha = cv2.threshold(gray_id, 1, 255, cv2.THRESH_BINARY)
                
                b, g, r = cv2.split(img)
                bgra = cv2.merge((b, g, r, alpha))
                
                dst = os.path.join(output_dir, "objects", f"{friendly_name}.png")
                cv2.imwrite(dst, bgra)
                
            for file in os.listdir(raw_dir):
                try:
                    os.remove(os.path.join(raw_dir, file))
                except:
                    pass
                
        # Destroy the object so the next one has a clean room
        c.communicate({"$type": "destroy_object", "id": obj_id})
    
    # Try to close the env nicely
    try:
        env.close()
    except:
        pass
    time.sleep(2)

    print("Generating room PNGs (Top-down view)...")
    scenes = [("1a", "tdw_mat/dataset/dataset_train/"), 
              ("2a", "tdw_mat/dataset/dataset_test/"), 
              ("4a", "tdw_mat/dataset/dataset_train/"), 
              ("5a", "tdw_mat/dataset/dataset_test/")]
    
    for scene, data_prefix in scenes:
        print(f"Loading scene {scene}...")
        # Launching the environment might try to launch TDW, let's also pass launch_build=False 
        # since TDW might be already running or we manually start it
        env = gym.make("transport_challenge_MA", port=1071, launch_build=False, data_prefix=data_prefix)
        env.reset(options={"scene": scene, "layout": "0_0", "task": "food", "seed": 0})
        
        output_path = os.path.join(output_dir, "rooms", f"{scene}.png")
        capture_topdown_room(env, output_path)
        print(f"Saved {scene}.png")
        
        try:
            env.close()
        except:
            pass
        time.sleep(2) # Wait a bit for the port to free up

    print("All done!")

if __name__ == "__main__":
    main()
