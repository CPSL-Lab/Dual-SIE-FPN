# Dual SIE-FPN: Semantic and Spatial Information Enhancement for Multi-Scale Object Detection

In this paper, we propose a novel semantic and spatial information enhancing feature pyramid network (**Dual SIE-FPN**), which mainly focuses on alleviating multi-scale hierarchical feature transmission loss and enhancing the feature representation. Specifically, **Dual SIE-FPN** contains three modules: Lateral Feature Enhancement (**LFE**), Global Attention Upsampling (**GAU**), and Multiple Information Compensation (**MIC**). **LFE** is designed to capture deep semantic representation and enhance channel information. **GAU** is established to make up for spatial information loss caused by upsampling, and transmit the high-level features with the compensatory information to low-level features simultaneously. **MIC** is designed to work with **LFE** in parallel to further improve the information loss resulting from 1×1 convolution.

## Development Environment

- Linux (tested on Ubuntu 22.04)
- detectron2
- CUDA 11.3
- CuDNN 8.2
- PyTorch 1.10.0
- Torchvision 0.11.0
- fvcore 0.1.5
- Pillow 8.2.0
- GCC 7.3

## Results on MS COCO test-dev2017

|        Backbone         |   detector   | lr schedule | mAP(BBox) | mAP(Segm) | model                                                        |
| :---------------------: | :----------: | :---------: | :-------: | :-------: | ------------------------------------------------------------ |
| ResNet-101/Dual SIE-FPN |  RetinaNet   |     1×      |   38.2    |     -     | [RetinaNet R50-Dual SIE-FPN](https://drive.google.com/file/d/1E3fQ8OvlcO32JFO3rMKHHvLSn8J4JcEW/view?usp=drive_link) |
| ResNet-101/Dual SIE-FPN |  RetinaNet   |     1×      |   40.1    |     -     | [RetinaNet R101-Dual SIE-FPN](https://drive.google.com/file/d/1lKdYqFGJVYKjBuAn9rv7JR5ozoKwt7Dz/view?usp=drive_link) |
| ResNet-50/Dual SIE-FPN  | Faster R-CNN |     1×      |   39.9    |     -     | [Faster R50-Dual SIE-FPN](https://drive.google.com/file/d/1soHYyg-R2Znaa9ti1hLv2qBwAtAmRTHB/view?usp=drive_link) |
| ResNet-101/Dual SIE-FPN | Faster R-CNN |     1×      |   41.9    |     -     | [Faster R101-Dual SIE-FPN](https://drive.google.com/file/d/1V_1o1Gqh1bul_zL68EUvhbVv8H5F913j/view?usp=drive_link) |
| ResNet-50/Dual SIE-FPN  |  Mask R-CNN  |     1×      |   40.9    |   37.0    | [Mask R50-Dual SIE-FPN](https://drive.google.com/file/d/1m6WdgFJvar-I0qYAlDFSzvRL63Y-4wqf/view?usp=drive_link) |
| ResNet-101/Dual SIE-FPN |  Mask R-CNN  |     1×      |   42.5    |   38.4    | [Mask R101-Dual SIE-FPN](https://drive.google.com/file/d/1S8ff6_LEYkIeeILEW7NX9mNXep12rfO5/view?usp=drive_link) |

## Training and Evaluation

To train a model, run

```
python ./train_net.py \
  --config-file ../configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml \
  --num-gpus <GPU_NUM> SOLVER.IMS_PER_BATCH <BATCH_SIZE> SOLVER.BASE_LR <LEARNINR RATE>
```

Model evaluation can be done similarly:

```
python ./train_net.py \
  --config-file ../configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml \
  --eval-only  MODEL.WEIGHTS /path/to/checkpoint_file
```

For more options, see `python ./train_net.py -h`.
