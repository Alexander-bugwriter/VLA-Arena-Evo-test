# Copyright 2025 The VLA-Arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
import xml.etree.ElementTree as ET

import robosuite
from huggingface_hub import hf_hub_download
from robosuite.utils.mjcf_utils import find_elements


DIR = os.path.dirname(__file__)

def apply_instruction_replacement(original_instruction: str, replacements_dict: dict, cfg, logger) -> str:
    """
    Apply random instruction replacement based on the replacements dictionary.
    
    Args:
        original_instruction: The original instruction string
        replacements_dict: Dictionary mapping normalized instructions to replacement lists
        cfg: Configuration object containing replacement settings
    
    Returns:
        The potentially replaced instruction string
    """
    if not cfg.use_replacements or not replacements_dict:
        return original_instruction
    
    # Check if we should apply replacement based on probability
    if random.random() > cfg.replacement_probability:
        return original_instruction
    
    # Convert instruction to key format: spaces to underscores, lowercase
    instruction_key = original_instruction.lower().replace(" ", "_")
    
    # Check if we have replacements for this instruction
    if instruction_key in replacements_dict:
        replacement_options = replacements_dict[instruction_key]
        if replacement_options:
            # Randomly select one replacement
            selected_replacement = random.choice(replacement_options)
            logger.info(f"Replaced instruction: '{original_instruction}' -> '{selected_replacement}'")
            return selected_replacement.replace("_", " ")
    
    # If no replacement found, return original instruction
    return original_instruction


def load_replacements_dict(cfg, logger) -> dict:
    """Load the replacements dictionary from JSON file."""
    if not cfg.use_replacements:
        return {}
    try:
        if cfg.replacements_file == 'VLA-Arena/language_replacements':
            filename = f"comprehensive_word_replacements_{cfg.replacement_level}.json"

            file_path = hf_hub_download(
                repo_id=cfg.replacements_file, 
                filename=filename,
                repo_type="dataset"
            )

            with open(file_path, 'r') as f:
                replacements_list = json.load(f)
        else:
            with open(cfg.replacements_file, 'r') as f:
                replacements_list = json.load(f)

        replacements_dict = {}
        for item in replacements_list:
            original_key = item.get('original')  # original instructions
            modified_value = item.get('modified')  # replaced instructions
            if original_key and modified_value:
                if original_key not in replacements_dict:
                    replacements_dict[original_key] = []
                replacements_dict[original_key].append(modified_value)
        
        logger.info(f"Loaded {len(replacements_dict)} replacement entries from {cfg.replacements_file}")
        return replacements_dict
        
    except FileNotFoundError:
        logger.info(f"Replacements file not found: {cfg.replacements_file}. Disabling replacements.")
        return {}
    except json.JSONDecodeError as e:
        logger.info(f"Error parsing replacements file: {e}. Disabling replacements.")
        return {}
    except Exception as e:
        logger.info(f"Unexpected error loading replacements: {e}. Disabling replacements.")
        return {}

def postprocess_model_xml(xml_str, cameras_dict={}):
    """
    This function postprocesses the model.xml collected from a MuJoCo demonstration
    in order to make sure that the STL files can be found.

    Args:
        xml_str (str): Mujoco sim demonstration XML file as string

    Returns:
        str: Post-processed xml file as string
    """

    path = os.path.split(robosuite.__file__)[0]
    path_split = path.split('/')

    # replace mesh and texture file paths
    tree = ET.fromstring(xml_str)
    root = tree
    asset = root.find('asset')
    meshes = asset.findall('mesh')
    textures = asset.findall('texture')
    all_elements = meshes + textures

    for elem in all_elements:
        old_path = elem.get('file')
        if old_path is None:
            continue
        old_path_split = old_path.split('/')
        if 'robosuite' not in old_path_split:
            continue
        ind = max(
            loc for loc, val in enumerate(old_path_split) if val == 'robosuite'
        )  # last occurrence index
        new_path_split = path_split + old_path_split[ind + 1 :]
        new_path = '/'.join(new_path_split)
        elem.set('file', new_path)

    # cameras = root.find("worldbody").findall("camera")
    cameras = find_elements(root=tree, tags='camera', return_first=False)
    for camera in cameras:
        camera_name = camera.get('name')
        if camera_name in cameras_dict:
            camera.set('name', camera_name)
            camera.set('pos', cameras_dict[camera_name]['pos'])
            camera.set('quat', cameras_dict[camera_name]['quat'])
            camera.set('mode', 'fixed')
    return ET.tostring(root, encoding='utf8').decode('utf8')


def process_image_input(img_tensor):
    # return (img_tensor / 255. - 0.5) * 2.
    return img_tensor / 255.0


def reconstruct_image_output(img_array):
    # return (img_array + 1.) / 2. * 255.
    return img_array * 255.0


def update_env_kwargs(env_kwargs, **kwargs):
    for k, v in kwargs.items():
        env_kwargs[k] = v
