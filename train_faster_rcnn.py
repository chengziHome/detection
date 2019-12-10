import torch
import torchvision
from pycocotools.coco import COCO

from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
# from torchvision.transforms import Compose
from torchvision.datasets import CocoDetection
from detection.transforms import ToTensor,RandomHorizontalFlip,Compose
from detection.utils import collate_fn
from detection.coco_utils import CocoDetection,get_coco
from datasets import COCODataset
from solver_detection import Solver




ROOT_train = "/home/jiazhaohe/datasets/COCO_2017/train2017"
ROOT_val   = "/home/jiazhaohe/datasets/COCO_2017/val2017"
ann_train = "/home/jiazhaohe/datasets/COCO_2017/annotations/instances_train2017.json"
ann_val   = "/home/jiazhaohe/datasets/COCO_2017/annotations/instances_val2017.json"
ROOT = "/home/jiazhaohe/datasets/COCO_2017"

coco_dataset_train = get_coco(root=ROOT,
                              image_set="train",
                                 transforms=Compose([
                                        ToTensor(),
                                        RandomHorizontalFlip(0.5)
                                 ]))
coco_dataset_val = get_coco(root=ROOT,
                            image_set="val",
                            transforms=ToTensor())

train_sampler = torch.utils.data.RandomSampler(coco_dataset_train)
val_sampler   = torch.utils.data.SequentialSampler(coco_dataset_val)
train_batch_sampler = torch.utils.data.BatchSampler(train_sampler,2,drop_last=True)

trainLoader = DataLoader(coco_dataset_train,batch_sampler=train_batch_sampler,
                          num_workers=4,collate_fn=collate_fn)
valLoader   = DataLoader(coco_dataset_val  ,batch_size=1,shuffle=False,
                         num_workers=2,collate_fn=collate_fn)

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

net = torchvision.models.resnet18(pretrained=True)
backbone = torch.nn.Sequential(net.conv1,net.bn1,net.relu,net.maxpool,
                               net.layer1,net.layer2,net.layer3,net.layer4)

backbone.out_channels = 512

anchor_generator = AnchorGenerator(sizes=((32,64,128,256,512)),
                                   aspect_ratios=((0.5,1.0,2.0)))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

model = FasterRCNN(backbone,
                   num_classes=91,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
model = model.cuda()



criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
solver = Solver(model,"./models/rcnn_4.pth",trainLoader,valLoader,criterion,optimizer,
                logfile="./logs/rcnn_resnet18.log",
                print_freq=20,save_name="rcnn")


solver.train(4)



