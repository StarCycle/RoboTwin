# Generate object tracking trajectories for every frame in the video
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
from pickle import dumps, loads
from PIL import Image
import lmdb
import numpy as np
import torch
from torchvision.io import encode_jpeg, decode_jpeg
import zlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', type=str, default='')
    parser.add_argument('--img_path', type=str, default='')
    # parser.add_argument('--chunk_size', type=int, default=50)
    # parser.add_argument('--track_num', type=int, default=50)
    parser.add_argument('--device', type=str, default="cuda:1")
    args = parser.parse_args()

    lmdb_path = Path(args.lmdb_path)
    img_path = Path(args.img_path)

    res = {
        "head_camera": 0, 
        "front_camera": 0, 
        "left_camera": 0, 
        "right_camera": 0,
    }
    max_steps = json.load(open(lmdb_path/'split.json', 'r'))
    split_num = len(max_steps)
    min_steps = [0] + [max_steps[split_id]+1 for split_id in range(split_num-1)]

    # cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)

    for split_id in range(split_num):
        env = lmdb.open(str(lmdb_path/str(split_id)), map_size=int(3e12))
        with env.begin() as txn:
            mask_ids = np.arange(7, 65) 
            for camera in res:
                res[camera] = loads(txn.get(f'{camera}_res'.encode()))
            for start_frame_id in range(min_steps[split_id], max_steps[split_id]):
                ep_id = txn.get(f'cur_episode_{start_frame_id}'.encode())

                # Aquire the mask of the manipulated object
                for camera in res:
                    seg = np.frombuffer(
                        zlib.decompress(
                            loads(txn.get(f'{camera}_seg_{start_frame_id}'.encode())),
                        ),
                        dtype=np.uint8,
                    ).reshape(res[camera])
                    bool_mask = np.isin(seg, mask_ids)
                    mask = np.where(bool_mask, 255, 0).astype(np.uint8)
                    mask_path = img_path / camera / 'mask' 
                    mask_path.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask).save(mask_path / f'{start_frame_id:05}.png')

                    rgb = decode_jpeg(
                        loads(txn.get(f'{camera}_rgb_{start_frame_id}'.encode())),
                    ).permute(1,2,0).numpy()
                    rgb_path = img_path / camera / 'rgb'
                    rgb_path.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb, 'RGB').save(rgb_path / f'{start_frame_id:05}.jpg')