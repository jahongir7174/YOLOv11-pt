YOLOv11 re-implementation using PyTorch

### Installation

```
conda create -n YOLO python=3.10.10
conda activate YOLO
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python
pip install PyYAML
pip install tqdm
```

### Train

* Configure your dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

### Results

| Version | Epochs | Box mAP |                                                                              Download |
|:-------:|:------:|--------:|--------------------------------------------------------------------------------------:|
|  v11_n  |  600   |    38.6 |                                                            [Model](./weights/best.pt) |
| v11_n*  |   -    |    39.2 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_n.pt) |
| v11_s*  |   -    |    46.5 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_s.pt) |
| v11_m*  |   -    |    51.2 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_m.pt) |
| v11_l*  |   -    |    53.0 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_l.pt) |
| v11_x*  |   -    |    54.3 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_x.pt) |

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.386
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.551
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.196
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.321
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.361
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.777
```

* `*` means that it is from original repository, see reference
* In the official YOLOv11 code, mask annotation information is used, which leads to higher performance

### Dataset structure

    ├── COCO 
        ├── images
            ├── train2017
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val2017
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train2017
                ├── 1111.txt
                ├── 2222.txt
            ├── val2017
                ├── 1111.txt
                ├── 2222.txt

#### Reference

* https://github.com/ultralytics/ultralytics
* https://github.com/jahongir7174/YOLOv8-pt
