# Read RGB images of different cameras from the pickle files and save gif
import os
import argparse
from pathlib import Path
import re
from tqdm import tqdm
import pickle
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

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
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()

    data_path = Path(args.path)
    num_episodes = count_episode_folders(data_path)

    for ep_id in tqdm(range(num_episodes), desc='visualizing episode'):
        videos = {}
        ep_path = data_path / f'episode{ep_id}'
        num_pkls = count_pkl_files(ep_path)
        for step_id in tqdm(range(num_pkls), desc='step'):
            with open(ep_path / f'{step_id}.pkl', 'rb') as f:
                data = pickle.load(f)
            for camera in data["observation"]:
                if camera not in videos:
                    videos[camera] = []
                    videos[camera+"_mask"] = []
                videos[camera].append(data["observation"][camera]['rgb'])
                videos[camera+"_mask"].append(data["observation"][camera]['actor_segmentation'])
        for camera in videos:        
            clip = ImageSequenceClip(videos[camera], fps=30)
            clip.write_gif(data_path / (f'{camera}_{ep_id}.gif'), fps=30)