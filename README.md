## Introduction

We introduce DistillNeRF, a generalizable model for 3D scene representation, self-supervised by natural sensor streams along with distillation from offline NeRFs and vision foundation models. It supports rendering RGB, depth, and foundation feature images, without test-time per-scene optimization, and enables downstream tasks such as zero-shot 3D semantic occupancy prediction and open-vocabulary text queries.

<p align="center">
  <img src="assets/overview.png" alt="DistillNeRF overview">
</p>

See more introductions and demos in this [Doc](PROJECT.md).

## News
- [ ] Release the code
- [ ] Release the trained model weight (coming soon🚀)
- [ ] Release the depths and RGB images used in the paper for distillation from offline per-scene NeRF (coming soon🚀)

## Installation

Our code is developed on Ubuntu 22.04 using Python 3.8 and PyTorch 1.13.1+cu116. Please note that the code has only been tested with these specified versions. We recommend using conda for the installation of dependencies.

Create the `distillnerf` conda **environment** and install all dependencies:

```shell
conda create -n distillnerf python=3.8 -y
conda activate distillnerf
. setup.sh
export PYTHONPATH=.
```


## NuScenes Dataset preparation

The NuScenes dataset is a popular autonomous driving dataset. Follow the steps below to set it up and prepare it for use:

1. **Download NuScenes Data**

The dataset can be downloaded from the NuScenes official website. Once downloaded, unzip it and place it at your favored place.

2. **Setup Directories**

Create directories for NuScenes data, and create a symbolic link to your saved Nuscenes data (symbolic link makes my life easier, I like it)

```shell
cd $DistillNeRF_Repo
mkdir -p data/nuscenes
ln -s $PATH_TO_NUSCENES data/nuscenes
```

3. **Preprocess data for mmdetection3d**
   
For NuScenes mini set, run
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini
```

For NuScenes full set, run
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0
```

1. **Download Sky Masks**

Sky masks help in addressing the ill-defined depth for sky. Download the sky masks from the provided Google Drive link (link coming soon after rebuttal for anonymous purpose) and unzip them in the `data/nuscenes/` directory.

1. **Prepare depth images for distillation**
   
   In this paper, we train offline per-scene NeRF, render depth images and save them to the `data/nuscenes/` directory. We'll release the depth images used in our paper soon. 
   
   In this repo, we also prepare some temporary data, so that at least you can run through the code. 
   To do that, download this Temporary File (link coming soon after rebuttal for anonymous purpose) and place it at the root directory of the repo. We have changed the `SKIP_MISSING` parameter in the dataset config (e.g. `project.configs.datasets.dataset_config.py`) to be True, so that the dataloader will load these temporary data files. When you start to train your model, turn `SKIP_MISSING` to be False, to avid data mis-load.

2. **Download some auxiliary models**

   Download Auxiliary Models (link coming soon after rebuttal for anonymous purpose) used during training the model (e.g. semantic segmentation for sky masks online generation)

Once done, you're all set to integrate and use the NuScenes dataset in your project!

## Run code

Here we provide scripts to visualize the data/predictions. If you're running the model locally with limited compute, you could append this line of argments after your script (after `--cfg-options`), so that the model only loads 1 camera instead of 6 cameras, which should be runnable in most machines.

```
model.num_camera=1 data.train.subselect_group_num=1 data.val.subselect_group_num=1 data.test.subselect_group_num=1
```

### Data Inspection

Before initiating the training, you might want to inspect the data and some initial predictions from our model. We've included scripts for visualizing the them.

1. **visualize images**

To run through the one DistillNeRF model (incorporating parameterized space and depth distillation), use

```
python tools/train.py ./project/configs/model_wrapper/model_wrapper.py --seed 0 --work-dir=../work_dir_debug --cfg-options model.visualize_imgs=True
```
Sample scripts for more models can be found in `sample_scripts/visualize_images`


1. **visualize voxels**
   
To run through the one DistillNeRF model (incorporating parameterized space and depth distillation), use

```****
python tools/train.py ./project/configs/model_wrapper/model_wrapper.py --seed 0 --work-dir=../work_dir_debug --cfg-options model.visualize_voxels=True
```
where we simply enable `visualize_voxels` to be True, instead of `visualize_images`. Sample scripts for more models can be found in `sample_scripts/visualize_voxels`



### Training

1. **Wandb setup (optional)**
   
Before start training, you may want to set up the wandb in order to log the metrics/predictions. 

You can run the script `wandb online` or `wandb offline` to choose wheter the logs will be uploaded to the cloud or saved locall. 

To set up your wandb account, you can follow the wandb prompt after you launch training. You can also uncomment these lines in `tools/train.py` and add you `WANDB_API_KEY` in advance.

```
os.environ["WANDB_API_KEY"] = 'YOUR_WANDB_API_KEY'
os.environ["WANDB_MODE"] = "online"
```

2. **Training script**
   
To run through one DistillNeRF model (incorporating parameterized space and depth distillation), use

```
python tools/train.py ./project/configs/model_wrapper/model_wrapper.py --seed 0 --work-dir=../work_dir_debug
```

For more examples, refer to the `sample_scripts/training`.



## Visualizations with trained models

- [ ] We will release all pretrained models soon, thus you can inspect the visualize predictions from our model.

After you obtain a trained model, we provide additional scripts to visualize the images and voxels, and also novel view synthesis. 

Running the commands below will save the visualization into a default directory. You can choose to not save the visualization by appending `model.save_visualized_imgs=False` to your command, and change the saving directory by appending `model.vis_save_directory=YOUR_VIS_DIR`.


1. **visualize images**

To run through one DistillNeRF model (incorporating parameterized space and depth distillation), use

```
python ./tools/visualization.py ./project/configs/model_wrapper/model_wrapper.py ./checkpoint/model.pth --cfg-options model.visualize_imgs=True
```

For more examples, refer to the `sample_scripts/visualization_images_with_model`.


2. **visualize voxels**
To run through one DistillNeRF model (incorporating parameterized space and depth distillation), use

```   
python ./tools/visualization.py ./project/configs/model_wrapper/model_wrapper.py ./checkpoint/model.pth --cfg-options model.visualize_voxels=True
```

For more examples, refer to the `sample_scripts/visualization_voxels_with_model`.

3. **foundation model feature visualization**

To visvualize DINO feature, run
```   
python ./tools/visualization.py ./project/configs/model_wrapper/model_wrapper_linearspace_dino.py ./checkpoint/model_linearspace_dino.pth --cfg-options model.visualize_foundation_model_feat=True 
```

To visvualize CLIP feature, run
```
python ./tools/visualization.py ./project/configs/model_wrapper/model_wrapper_linearspace_clip.py ./checkpoint/model_linearspace_clip.pth --cfg-options model.visualize_foundation_model_feat=True 
```

4. **open-vocabulary query**

To conduct open-vocabulary query, use

```   
python ./tools/visualization.py ./project/configs/model_wrapper/model_wrapper_linearspace_clip.py ./checkpoint/model_linearspace_clip.pth --cfg-options model.language_query=True 
```

5. **novel-view synthesis - RGB**

To run through one DistillNeRF model (incorporating depth distillation), use

```
python ./tools/novel_view_synthesis.py ./project/configs/model_wrapper/model_wrapper.py ./checkpoint/model**.pth** --cfg-options model.visualize_imgs=True
```

To run through one DistillNeRF model (incorporating depth distillation, and virtual camera distillation), use
```
python ./tools/novel_view_synthesis.py ./project/configs/model_wrapper/model_wrapper_linearspace.py ./checkpoint/model_linearspace_virtual_cam.pth --cfg-options model.visualize_imgs=True
```

The scripts above will generate 3 novel views. To generate more novel views and create a video, use this command
```
. ./tools/novel_view_synthesis.sh
```
Note that you need to choose with model you want to use, by commenting and uncommenting in `./tools/novel_view_synthesis.sh`.


6. **novel-view synthesis - foundation model feature**

To generate the novel view of DINO feature, use

```
    python ./tools/novel_view_synthesis.py ./project/configs/model_wrapper/model_wrapper_linearspace_dino.py ./checkpoint/model_linearspace_dino.pth --cfg-options model.visualize_foundation_model_feat=True 
```

To generate the novel view of CLIP feature, use

```
    python ./tools/novel_view_synthesis.py ./project/configs/model_wrapper/model_wrapper_linearspace_clip.py ./checkpoint/model_linearspace_clip.pth --cfg-options model.visualize_foundation_model_feat=True 
```

Again, the scripts above will generate 3 novel views. To generate more novel views and create a video, use this command
```
. ./tools/novel_view_synthesis.sh
```
Note that you need to choose with model you want to use, by commenting and uncommenting in `./tools/novel_view_synthesis.sh`.


## Acknowledgement

This implementation is based on the following codebase. Thanks for the great works!

- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
- [SDFStudio](https://github.com/autonomousvision/sdfstudio)

## Licence

### All Rights Reserved

Unauthorized copying of this file, via any medium, is strictly prohibited.

### Proprietary and Confidential

This file is proprietary and confidential. Unauthorized copying, distribution, or use of this file, via any medium, is strictly prohibited.

### External Licence
Note that one of our dependencies, `lyft-dataset-sdk`, is under a more restrictive non-commercial license. Usage of this file must comply with the non-commercial terms of the lyft-dataset-sdk license. We would consider remove this dependency later.