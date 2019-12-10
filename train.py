import datasets
import solver
import os

import torch
import torchvision
import torchvision.transforms as transforms


ROOT = "/home/jiazhaohe/datasets/imagenet12"

meta = torch.load(os.path.join(ROOT,"meta_data.bin"))
train_samples = meta["train_samples"]
val_samples = meta["val_samples"]

transform = torchvision.transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            ])

trainset = datasets.ImgListDataset(os.path.join(ROOT,"train"),transform,train_samples)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=256,
                                          shuffle=True,num_workers=16,drop_last=True)

valset = datasets.ImgListDataset(os.path.join(ROOT,"val"),transform,val_samples)
valloader = torch.utils.data.DataLoader(valset,batch_size=256,
                                          shuffle=False,num_workers=16,drop_last=True)


net = torchvision.models.alexnet()
net = net.cuda()

criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(),lr=0.05,momentum=0.9,weight_decay=5e-4)

mySolver = solver.Solver(net=net,check_point="./models/best_5.pth",
                        trainloader=trainloader,valloader=valloader,
                        criterion=criterion,optimizer=optimizer,
                        logfile="./logs/alexnet.log",
                        print_freq=20,
                        save_name="alexnet")
# lr=0.05
#mySolver.train(3)

optimizer = torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.9,weight_decay=5e-4)
mySolver.optimizer = optimizer
mySolver.train(12)

optimizer = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9,weight_decay=5e-4)
mySolver.optimizer = optimizer
mySolver.train(15)


optimizer = torch.optim.SGD(net.parameters(),lr=0.0001,momentum=0.9,weight_decay=5e-4)
mySolver.optimizer = optimizer
mySolver.train(10)






