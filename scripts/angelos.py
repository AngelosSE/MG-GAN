from mggan.data_utils.data_loaders import get_dataloader
from mggan.model.train import PiNetMultiGeneratorGAN
from mggan.evaluation import to_numpy, plot_trajectories_by_idxs,get_same_obs_indices,plot_trajectories
import numpy as np
from mggan.manifold import Manifold
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('/Users/angtoy/Box Sync/Research/Notebook/Documents/cache_tools(9feb2023)')
from cache_tools import cache_calls_on_disk
import pandas as pd
import os
from PIL import Image
from mggan.data_utils.experiments import stanford as Stanford
import argparse



def load_image(path_to_image,scene_name,dataset_name,experiment):
    
    img = Image.open(path_to_image)
    if "stanford" in dataset_name or "gofp" in dataset_name:
        if "stanford" in dataset_name:
            ratio = experiment['homography'].loc[
                (
                    (experiment['homography']["File"] == "{}.jpg".format(scene_name))
                    & (experiment['homography']["Version"] == "A")
                ),
                "Ratio",
            ].iloc[0]
        elif "gofp" in dataset_name:
            ratio = experiment['homography'][scene_name]

        scale_factor = ratio / experiment['img_scaling']

        old_width = img.size[0]
        old_height = img.size[1]

        new_width = int(round(old_width * scale_factor))
        new_height = int(round(old_height * scale_factor))

        scaled_img = img.resize((new_width, new_height), Image.ANTIALIAS)
    else:
        scaled_img = img
        scale_factor = 1
        ratio = 1.0

    return np.array(scaled_img)

parser = argparse.ArgumentParser()
parser.add_argument('n')

def main():
    model_dir = Path('/Users/angtoy/Documents/MG-GAN/checkpoints/sdd/version_5')
    phase = 'test'
    split= 'all'
    pred_strat='smart_expected' #'sampling'
    num_preds = 20
    num_preds_list = list(range(1, num_preds))
    model, config = PiNetMultiGeneratorGAN.load_from_path(model_dir, "best")
    test_loader = get_dataloader(
        config.dataset, phase, batch_size=32, split=split
        )
    model.get_predictions = cache_calls_on_disk('/Users/angtoy/Documents/MG-GAN/cache/')(model.get_predictions)
    preds = model.get_predictions(
        test_loader, max(num_preds_list), strategy=pred_strat
        )
    preds = preds.transpose(2, 1, 0, 3)
    gt_trajs = to_numpy(test_loader.dataset.pred_traj)
    
    obs_trajs = to_numpy(test_loader.dataset.obs_traj)
    same_scenes_indices,scenes = get_same_obs_indices(test_loader.dataset)
    idx_scene = 25*int(parser.parse_args().n) # 2500 looks odd, 220 is a nice failure, 70*7, 70*16, 70*44
    print(f'{len(same_scenes_indices)=}')
    same_scene_indices = same_scenes_indices[idx_scene]
    ped_index = 0
    same_ped_indices = list(zip(*same_scene_indices))[ped_index]
    print(f'{len(same_ped_indices)=}')
    same_ped_indices = np.array(same_ped_indices)
    pred_mask = np.isnan(gt_trajs).any(-1).any(-1)
    not_pred_mask_indices = np.where(~pred_mask)[0]
    same_ped_indices = np.intersect1d(same_ped_indices, not_pred_mask_indices)
    cur_preds = preds[same_ped_indices].reshape(-1, *preds.shape[2:])
    #gt_man_samples = gt_trajs[same_ped_indices]
    #gt_man = Manifold(gt_man_samples, radius=3)
    #inside_man = gt_man.compute_inside(cur_preds)
    traj_idx = same_ped_indices[0]

#    img = np.array(test_loader.dataset[traj_idx][4][0]['scaled_image'])
    scene = scenes[idx_scene]
    path_to_image = f'/Users/angtoy/Documents/MG-GAN/data/datasets/stanford/test/{scene}.jpg'
    img = load_image(path_to_image,scene,'stanford',Stanford().args_dict)

    plot_trajectories(obs_trajs[traj_idx], gt_trajs[traj_idx], cur_preds,img=img)
    #plot_trajectories_by_idxs(
    #    obs_trajs[traj_idx], None, cur_preds)#, inside_man.astype(np.int)
        #)
    
    plt.show()

if __name__ == '__main__':
    main()