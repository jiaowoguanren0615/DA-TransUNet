<h1 align="center">DA-TransUNet</h1>

## [DA-TransUNet: Integrating Spatial and Channel Dual Attention with Transformer U-Net for Medical Image Segmentation](https://arxiv.org/abs/2310.12570)

This is a warehouse for DA-TransUNet-pytorch-model, can be used to train your image datasets for segmentation tasks.
The code mainly comes from official [source code](https://github.com/SUN-1024/DA-TransUnet).

## Project Structure
```
├── datasets: Load datasets
    ├── cityscapes.py: Customize reading CityScapes dataset
    ├── coco.py: Customize reading COCO dataset
    ├── voc.py: Customize reading PascalVOC dataset
├── models: DA-TransUNet Model
    ├── block.py: Build the necessary branches of the model
    ├── build_model.py: Construct "DA-TransUNet" model
    ├── configs.py: model config parameters
├── utils:
    ├── augmentations.py: Define transforms data enhancement methods
    ├── distributed_utils.py: Record various indicator information and output and distributed environment
    ├── losses.py: Define loss functions, focal_loss, ce_loss, Dice, etc
    ├── metrics.py: Compute iou, f1, pixel_accuracy.
    ├── optimizer.py: Get a optimizer (AdamW or SGD)
    ├── schedulers.py: Define lr_schedulers (PolyLR, WarmupLR, WarmupPolyLR, WarmupExpLR, WarmupCosineLR, etc)
    ├── utils.py: Define some support functions(fix random seed, get model size, get time, throughput, etc)
    ├── visualize.py: Visualize datasets and predictions
├── engine.py: Function code for a training/validation process
└── train_gpu.py: Training model startup file
```

## Precautions
This code repository is a reproduction of the ___DA-TransUNet___ model based on the Pytorch framework. If you need to train your own dataset, please add a customized Mydataset in the ___datasets___ directory. class, and then modify the ___data_root___, ___batchsize___, ___epochs___ and other training parameters in the ___train_gpu.py___ file.


## Train this model
### Parameters Meaning:
```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```
### Note: 
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1 on each GPU___, otherwise BN layer maybe report an error. If you recive an error like "___error: unrecognized arguments: --local-rank=1___" when you use distributed multi-GPUs training, just replace the command "___torch.distributed.launch___" to "___torch.distributed.run___".

### train model with single-machine single-GPU：
```
python train_gpu.py
```

### train model with single-machine multi-GPU：
```
python -m torch.distributed.launch --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU: 
(using a specified part of the GPUs: for example, I want to use the second and fourth GPUs)
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain GPU, just add CUDA_VISIBLE_DEVICES= to specify the index number of the GPU before each command. The principle is the same as single-machine multi-GPU training)
```
On the first machine: python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## Citation
```
@article{sun2023transunet,
  title={DA-TransUNet: Integrating Spatial and Channel Dual Attention with Transformer U-Net for Medical Image Segmentation},
  author={Sun, Guanqun and Pan, Yizhi and Kong, Weikun and Xu, Zichang and Ma, Jianhua and Racharak, Teeradaj and Nguyen, Le-Minh},
  journal={arXiv preprint arXiv:2310.12570},
  year={2023}
}
```

