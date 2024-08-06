# w\o parameterization, w\ depth disllation, w\o virtual cam distillation
python ./tools/visualization.py ./project/configs/model_wrapper/model_wrapper.py ./checkpoint/update_v2_0607_epoch_135.pth --cfg-options model.visualize_imgs=True

# w\ parameterization, w\ depth disllation, w\o virtual cam distillation
python ./tools/visualization.py ./project/configs/model_wrapper/model_wrapper_linearspace.py ./checkpoint/model_linearspace.pth --cfg-options model.visualize_imgs=True

# w\ parameterization, w\ depth disllation, w\ virtual cam distillation
python ./tools/visualization.py ./project/configs/model_wrapper/model_wrapper_linearspace.py ./checkpoint/update_v2_linearspace_virtual_cam_0513_epoch_188.pth --cfg-options model.visualize_imgs=True