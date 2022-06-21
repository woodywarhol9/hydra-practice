# MNIST Classification
### Summary
- Pytorch Lightning으로 MNIST Classification 수행
- [hydra](https://hydra.cc/)로 configurations 모듈화하여 관리
###  Project Structure
---
``` bash
├── configs
│   ├── checkpoint
│   │   └── checkpoint.yaml
│   ├── config.yaml
│   ├── data
│   │   └── mnist.yaml
│   ├── early_stopping
│   │   └── early_stopping.yaml
│   ├── logger
│   │   └── tensorboard.yaml
│   ├── lr_monitor
│   │   └── lr_monitor.yaml
│   ├── model
│   │   ├── resnet18.yaml
│   │   └── resnet34.yaml
│   ├── optimizer
│   │   └── sgd.yaml
│   ├── scheduler
│   │   └── steplr.yaml
│   └── trainer
│       └── trainer.yaml
├── lit_model.py
├── MNIST
├── outputs
│   └── 2022-06-20
│       └── 13-53-43
│           ├── epoch=5-val_loss=0.05-val_acc=0.99.ckpt
│           ├── mnist_classifier
│           │   └── version_0
│           │       ├── events.out.tfevents.1655700828.wjs-desktop.5467.0
│           │       ├── events.out.tfevents.1655700977.wjs-desktop.5467.1
│           │       └── hparams.yaml
│           └── trainer.log
├── README.md
├── requirements.txt
└── trainer.py
```
### Requirments
---
```
hydra-core==1.2.0
lightning_bolts==0.5.0
omegaconf==2.2.2
pytorch_lightning==1.6.4
torch==1.11.0+cu113
torchmetrics==0.9.1
torchvision==0.12.0+cu113
```
### Configurations
- config.yaml
---
```
seed : 42
monitor : "val_loss"

defaults:
  - model: resnet18
  - data : mnist
  - optimizer : sgd
  - scheduler : steplr
  - trainer : trainer
  - logger : tensorboard
  - early_stopping : early_stopping
  - lr_monitor : lr_monitor
  - checkpoint : checkpoint
```


### How to Run
- yaml 파일의 configs 지정 가능
---
``` bash
python trainer.py # use defaults hydra config
python trainer.py model=resnet34 # change resnet18 to resnet34
```

### Check log data
- `outputs/날짜/시간/`에 저장
    - configurations 정보 : .hydra/config.yaml
    - log data : mnist_classifier
---
```
tensorboard --logdir=outputs/2022-06-20/13-53-43/mnist_classifier
```
### Result
- resnet18
- 5 epoch에서 최고 성능  

Metrics|Validation|Test
:----|:--------|:----|
Accuracy|0.9869|0.986

![image](https://user-images.githubusercontent.com/86637320/174533344-d8fb21f0-85eb-4673-8a9f-73061d92796f.png)![image](https://user-images.githubusercontent.com/86637320/174539608-e9ca4813-6830-472f-a659-395ca15bff80.png)
