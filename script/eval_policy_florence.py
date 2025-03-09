
import sys
sys.path.append('./') 

import torch  
import os
import numpy as np
import hydra
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

def class_decorator(task_name):
    envs_module = importlib.import_module(f'envs.{task_name}')
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, '../task_config/_camera_config.yml')

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f'camera {camera_type} is not defined'
    return args[camera_type]

def main(usr_args):
    task_name = usr_args.task_name
    head_camera_type = usr_args.head_camera_type
    seed = usr_args.seed

    with open(f'./task_config/{task_name}.yml', 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args['head_camera_type'] = head_camera_type
    head_camera_config = get_camera_config(args['head_camera_type'])
    args['head_camera_fovy'] = head_camera_config['fovy']
    args['head_camera_w'] = head_camera_config['w']
    args['head_camera_h'] = head_camera_config['h']
    head_camera_config = 'fovy' + str(args['head_camera_fovy']) + '_w' + str(args['head_camera_w']) + '_h' + str(args['head_camera_h'])
    
    wrist_camera_config = get_camera_config(args['wrist_camera_type'])
    args['wrist_camera_fovy'] = wrist_camera_config['fovy']
    args['wrist_camera_w'] = wrist_camera_config['w']
    args['wrist_camera_h'] = wrist_camera_config['h']
    wrist_camera_config = 'fovy' + str(args['wrist_camera_fovy']) + '_w' + str(args['wrist_camera_w']) + '_h' + str(args['wrist_camera_h'])

    front_camera_config = get_camera_config(args['front_camera_type'])
    args['front_camera_fovy'] = front_camera_config['fovy']
    args['front_camera_w'] = front_camera_config['w']
    args['front_camera_h'] = front_camera_config['h']
    front_camera_config = 'fovy' + str(args['front_camera_fovy']) + '_w' + str(args['front_camera_w']) + '_h' + str(args['front_camera_h'])

    # output camera config
    print('============= Camera Config =============\n')
    print('Head Camera Config:\n    type: '+ str(args['head_camera_type']) + '\n    fovy: ' + str(args['head_camera_fovy']) + '\n    camera_w: ' + str(args['head_camera_w']) + '\n    camera_h: ' + str(args['head_camera_h']))
    print('Wrist Camera Config:\n    type: '+ str(args['wrist_camera_type']) + '\n    fovy: ' + str(args['wrist_camera_fovy']) + '\n    camera_w: ' + str(args['wrist_camera_w']) + '\n    camera_h: ' + str(args['wrist_camera_h']))
    print('Front Camera Config:\n    type: '+ str(args['front_camera_type']) + '\n    fovy: ' + str(args['front_camera_fovy']) + '\n    camera_w: ' + str(args['front_camera_w']) + '\n    camera_h: ' + str(args['front_camera_h']))
    print('\n=======================================')

    task = class_decorator(args['task_name'])

    st_seed = 100000 * (1+seed)
    suc_nums = []
    test_num = 100
    topk = 1

    # Load florence policy
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from mimictest.Utils.PreProcess import PreProcess
    from mimictest.Datasets.RoboTwinLMDBDataset import RoboTwinLMDBDataset
    from mimictest.Wrappers.DiffusionPolicy import DiffusionPolicy
    from mimictest.Simulation.SequentialRoboTwin import RoboTwinPolicy
    from mimictest.Nets.FlorencePi0Net import FlorencePi0Net

    # Saving path
    save_path = Path('/home/lizhuoheng/RoboTwin/policy/mimictest/mimictest/Scripts/RoboTwinExperiments/Save')
    save_path.mkdir(parents=True, exist_ok=True)
    load_batch_id = 2900

    # Dataset
    folder_name = 'lmdb_50ep_blockhammerbeat'
    dataset_path = Path('/home/lizhuoheng/RoboTwin/data') / folder_name

    # Space
    num_actions = 14
    lowdim_obs_dim = 14
    obs_horizon = 1
    chunk_size = 8
    process_configs = {
        'rgb': {
            'img_shape': (320, 320), # Initial resolution is (180, 320)
            'crop_shape': (280, 280),
            'max': torch.tensor(1.0),
            'min': torch.tensor(0.0),
        },
        'coord': {},
        'low_dim': {
            'max': None, # to be filled
            'min': None,
        },
        'action': {
            'max': None, # to be filled
            'min': None,
        },
        'mask': {},
    }
    loss_configs = {
        'action': {
            'loss_func': torch.nn.functional.l1_loss,
            'type': 'flow',
            'weight': 1.0,
            'shape': (chunk_size, num_actions),
        },
    }

    # Network
    model_path = Path("microsoft/Florence-2-base")
    freeze_vision_tower = True
    freeze_florence = False
    do_compile = False
    do_profile = False

    # Diffusion
    diffuser_train_steps = 10
    diffuser_infer_steps = 10
    diffuser_solver = "flow_euler"
    beta_schedule = None
    prediction_type = None
    clip_sample = None
    ema_interval = 10

    # Preparation
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        kwargs_handlers=[kwargs],
        mixed_precision='bf16',
    )
    device = acc.device
    train_dataset = RoboTwinLMDBDataset(
        dataset_path=dataset_path, 
        obs_horizon=obs_horizon, 
        chunk_size=chunk_size, 
        start_ratio=0,
        end_ratio=0.9,
    )
    limit = train_dataset.get_pos_range()
    process_configs['low_dim']['max'] = limit['pos_max']
    process_configs['low_dim']['min'] = limit['pos_min']
    process_configs['action']['max'] = limit['pos_max']
    process_configs['action']['min'] = limit['pos_min']
    preprocessor = PreProcess(
        process_configs=process_configs,
        device=device,
    )
    net = FlorencePi0Net(
        path=model_path,
        freeze_vision_tower=freeze_vision_tower,
        num_actions=num_actions,
        lowdim_obs_dim=lowdim_obs_dim,
    ).to(device)
    policy = DiffusionPolicy(
        net=net,
        loss_configs=loss_configs,
        do_compile=do_compile,
        scheduler_name=diffuser_solver,
        num_train_steps=diffuser_train_steps,
        num_infer_steps=diffuser_infer_steps,
        ema_interval=ema_interval,
        beta_schedule=beta_schedule,
        clip_sample=clip_sample,
        prediction_type=prediction_type,
    )
    policy.load_pretrained(acc, save_path, load_batch_id)
    policy.net, policy.ema_net = acc.prepare(
        policy.net, 
        policy.ema_net, 
        device_placement=[True, True],
    )
    if policy.use_ema:
        policy.ema_net.eval()
    else:
        policy.net.eval()
    policy = RoboTwinPolicy(policy, preprocessor, obs_horizon, chunk_size, save_path, record_video=True)

    st_seed, suc_num = test_policy(task, args, policy, st_seed, test_num=test_num)

    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]
    save_dir = Path(f'result_florence/{task_name}_{usr_args.head_camera_type}_{usr_args.expert_data_num}') 
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f'ckpt_{load_batch_id}_seed_{seed}.txt'
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(file_path, 'w') as file:
        file.write(f'Timestamp: {current_time}\n\n')

        file.write(f'Checkpoint Num: {load_batch_id}\n')
        
        file.write('Successful Rate of Diffenent checkpoints:\n')
        file.write('\n'.join(map(str, np.array(suc_nums) / test_num)))
        file.write('\n\n')
        file.write(f'TopK {topk} Success Rate (every):\n')
        file.write('\n'.join(map(str, np.array(topk_success_rate) / test_num)))
        file.write('\n\n')
        file.write(f'TopK {topk} Success Rate:\n')
        file.write(f'\n'.join(map(str, np.array(topk_success_rate) / (topk * test_num))))
        file.write('\n\n')

    print(f'Data has been saved to {file_path}')
    

def test_policy(Demo_class, args, policy, st_seed, test_num=20):
    expert_check = True

    Demo_class.suc = 0
    Demo_class.test_num =0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    

    now_seed = st_seed
    while succ_seed < test_num:
        render_freq = args['render_freq']
        args['render_freq'] = 0
        
        if expert_check:
            try:
                Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
                Demo_class.play_once()
                Demo_class.close()
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(' -------------')
                print('Error: ', stack_trace)
                print(' -------------')
                Demo_class.close()
                now_seed += 1
                args['render_freq'] = render_freq
                print('error occurs !')
                continue

        if (not expert_check) or ( Demo_class.plan_success and Demo_class.check_success() ):
            succ_seed +=1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args['render_freq'] = render_freq
            continue

        args['render_freq'] = render_freq

        Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
        Demo_class.apply_florence(policy)

        now_id += 1
        Demo_class.close()
        if Demo_class.render_freq:
            Demo_class.viewer.close()
        policy.log_video(succ_seed)
        policy.reset_obs()
        print(f"success rate: {Demo_class.suc}/{Demo_class.test_num}, current seed: {now_seed}\n")
        Demo_class._take_picture()
        now_seed += 1

    return now_seed, Demo_class.suc

if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()
    
    parser = ArgumentParser()

    # Add arguments
    parser.add_argument('--task_name', type=str, default='block_hammer_beat')
    parser.add_argument('--head_camera_type', type=str, default='L515')
    parser.add_argument('--seed', type=int, default=0)
    usr_args = parser.parse_args()
    
    main(usr_args)
