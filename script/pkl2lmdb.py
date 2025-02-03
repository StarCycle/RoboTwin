# Transfer pkl dataset to lmdb dataset
import os
import argparse
import json
from pathlib import Path
import re
from tqdm import tqdm
import pickle
from pickle import dumps
import lmdb
import torch
from torchvision.io import encode_jpeg
import zlib

def count_episode_folders(path):
    # Traverse the path to count there are how many folders with name 'episodex'
    count = 0
    pattern = re.compile(r'^episode\d+$')
    
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path) and pattern.match(item):
            count += 1
    return count

def count_pkl_files(path):
    count = 0
    pattern = re.compile(r'^\d+\.pkl$')
    
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isfile(full_path) and pattern.match(item):
            count += 1
    return count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=str, default='')
    parser.add_argument('--lmdb_path', type=str, default='')
    args = parser.parse_args()

    pkl_path = Path(args.pkl_path)
    lmdb_path = Path(args.lmdb_path)
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)

    num_episodes = count_episode_folders(pkl_path)
    single_file_episode_num = 1000
    step_in_lmdb = 0
    max_steps = []
    split_id = -1

    for ep_id in tqdm(range(num_episodes), desc='saving episode'):
        ep_path = pkl_path / f'episode{ep_id}'
        if ep_id % single_file_episode_num == 0:
            split_id += 1
            if ep_id != 0:
                max_steps.append(step_in_lmdb)
                json.dump(max_steps, open(lmdb_path/'split.json', 'w'))
                txn.commit()
                env.close()
            env = lmdb.open(str(lmdb_path/str(split_id)), map_size=int(3e12), readonly=False, lock=False) # maximum size of memory map is 3TB
            txn = env.begin(write=True)
        num_pkls = count_pkl_files(ep_path)
        for step_id in tqdm(range(num_pkls), desc='step'):
            with open(ep_path / f'{step_id}.pkl', 'rb') as f:
                data = pickle.load(f)
            for camera in data['observation']:
                rgb = encode_jpeg(torch.from_numpy(data['observation'][camera]['rgb']).permute(2,0,1))
                if ep_id == 0 and step_id == 0:
                    txn.put(f'{camera}_res'.encode(), dumps(data['observation'][camera]['depth'].shape))
                depth = zlib.compress(data['observation'][camera]['depth'].tobytes())
                seg = zlib.compress(data['observation'][camera]['actor_segmentation'].tobytes())
                intrinsic_cv = data['observation'][camera]['intrinsic_cv']
                extrinsic_cv = data['observation'][camera]['extrinsic_cv']
                cam2world_gl = data['observation'][camera]['cam2world_gl']
                txn.put(f'{camera}_rgb_{step_in_lmdb}'.encode(), dumps(rgb))
                txn.put(f'{camera}_depth_{step_in_lmdb}'.encode(), dumps(depth))
                txn.put(f'{camera}_seg_{step_in_lmdb}'.encode(), dumps(seg))
                txn.put(f'{camera}_intrinsic_cv_{step_in_lmdb}'.encode(), dumps(intrinsic_cv))
                txn.put(f'{camera}_extrinsic_cv_{step_in_lmdb}'.encode(), dumps(extrinsic_cv))
                txn.put(f'{camera}_cam2world_gl_{step_in_lmdb}'.encode(), dumps(cam2world_gl))
            pcd = data['pointcloud']
            txn.put(f'pointcloud_{step_in_lmdb}'.encode(), dumps(pcd))
            joint_action = data['joint_action']
            txn.put(f'joint_action_{step_in_lmdb}'.encode(), dumps(joint_action))
            endpose = data['endpose']
            txn.put(f'endpose_{step_in_lmdb}'.encode(), dumps(endpose))
            txn.put(f'cur_episode_{step_in_lmdb}'.encode(), dumps(ep_id))
            txn.put(f'cur_step_{step_in_lmdb}'.encode(), dumps(step_in_lmdb))
            step_in_lmdb += 1
    max_steps.append(step_in_lmdb)
    json.dump(max_steps, open(lmdb_path/'split.json', 'w'))
    txn.commit()
    env.close()
    import pdb; pdb.set_trace()