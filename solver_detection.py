
import logging
import os
import torch
import shutil
import math

from detection.coco_utils import get_coco_api_from_dataset
from detection.coco_eval import CocoEvaluator


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
        

        

class Solver():
    
    def __init__(self,net,check_point,trainloader,valloader,criterion,optimizer,
                logfile,print_freq,save_name):
        if not net:
            raise ValueError("We need net arch.So the param 'net' cannot be null.")
        
        # log
        self.logger = logging.getLogger(logfile)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(logfile)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
   
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.print_freq = print_freq
        self.save_name = save_name
        
        # statistics data
        self.net = net
        if check_point:
            self.load_checkpoint(check_point)
        else:
            # reset
            self.epoch = 0  # base on 1,not 0
            # loss per itertation
            self.train_loss_classifier = []
            self.train_loss_box_reg = []
            self.train_loss_objectness = []
            self.train_loss_rpn_box_reg = []
            self.train_loss_sum = []
            
            # loss per epoch
            self.train_epoch_loss_classifier = []
            self.train_epoch_loss_box_reg = []
            self.train_epoch_loss_objectness = []
            self.train_epoch_loss_rpn_box_reg = []
            self.train_epoch_loss_sum = []
            

            
            # coco val
            self.AP_50_95 = []
            self.AP_50 = []
            self.AP_75 = []
            self.AP_small = []
            self.AP_medium = []
            self.AP_large = []
            self.AR_1 = []
            self.AR_10 = []
            self.AR_100 = []
            self.AR_small = []
            self.AR_medium = []
            self.AR_large = []
            
        torch.backends.cudnn.benchmark = True
        
        self.logger.info("Solver init done...")
    
    def load_checkpoint(self,check_point):
        self.logger.info("Load checkpoint from "+check_point)
        data = torch.load(check_point)
        self.net.load_state_dict(data["state_dict"])
        self.epoch = data["epoch"]

        # loss per itertation
        self.train_loss_classifier = data["train_loss_classifier"]
        self.train_loss_box_reg = data["train_loss_box_reg"]
        self.train_loss_objectness = data["train_loss_objectness"]
        self.train_loss_rpn_box_reg = data["train_loss_rpn_box_reg"]
        self.train_loss_sum = data["train_loss_sum"]

        # loss per epoch
        self.train_epoch_loss_classifier = data["train_epoch_loss_classifier"]
        self.train_epoch_loss_box_reg = data["train_epoch_loss_box_reg"]
        self.train_epoch_loss_objectness = data["train_epoch_loss_objectness"]
        self.train_epoch_loss_rpn_box_reg = data["train_epoch_loss_rpn_box_reg"]
        self.train_epoch_loss_sum = data["train_epoch_loss_sum"]

        # coco val
        self.AP_50_95 = data["AP_50_95"]
        self.AP_50 = data["AP_50"]
        self.AP_75 = data["AP_75"]
        self.AP_small = data["AP_small"]
        self.AP_medium = data["AP_medium"]
        self.AP_large = data["AP_large"]
        self.AR_1 = data["AR_1"]
        self.AR_10 = data["AR_10"]
        self.AR_100 = data["AR_100"]
        self.AR_small = data["AR_small"]
        self.AR_medium = data["AR_medium"]
        self.AR_large = data["AR_large"]
        
        self.logger.info("Load checkpoint done,current epoch:"+str(self.epoch))
        
  
    
    
    def train(self,epochs):
        self.logger.info("begin training from epoch="+str(self.epoch+1))
#         iter_nums = int(math.ceil(len(self.trainloader.dataset)/self.trainloader.batch_size))
        iter_nums = len(self.trainloader)
        self.logger.info("There are "+ str(iter_nums) +" iterations per epoch.")
        for epoch in range(epochs):
            self.net.train()
            self.epoch = self.epoch + 1
            loss_all = AverageMeter()
            loss_classifier = AverageMeter()
            loss_box_reg = AverageMeter()
            loss_objectness = AverageMeter()
            loss_rpn_box_reg = AverageMeter()
            
            
            
            for i,(inputs,targets) in enumerate(self.trainloader):
                
                inputs = list(img.cuda() for img in inputs)
                targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
                
                loss_dict = self.net(inputs,targets)
                
                losses = sum(loss for loss in loss_dict.values())
             
                # measure and record
                self.train_loss_classifier.append(loss_dict["loss_classifier"].item())
                self.train_loss_box_reg.append(loss_dict["loss_box_reg"].item())
                self.train_loss_objectness.append(loss_dict["loss_objectness"].item())
                self.train_loss_rpn_box_reg.append(loss_dict["loss_rpn_box_reg"].item())
                self.train_loss_sum.append(losses.item())              
                
                loss_classifier.update(loss_dict["loss_classifier"].item())
                loss_box_reg.update(loss_dict["loss_box_reg"].item())
                loss_objectness.update(loss_dict["loss_objectness"].item())
                loss_rpn_box_reg.update(loss_dict["loss_rpn_box_reg"].item())
                loss_all.update(losses.item()) 
                
                # BP
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                
                
                if i%self.print_freq == (self.print_freq-1):
                    self.logger.info(
                        "[{epoch:2d},{iter:5d}/{num_iter:d}] "
                        "loss:({loss.val:.5f},{loss.avg:.5f})  "
                        "loss_cls:({loss_cls.val:6.3f},{loss_cls.avg:6.3f})  "
                        "loss_box_reg:({loss_box_reg.val:6.3f},{loss_box_reg.avg:6.3f})  "
                        "loss_objectness:({loss_objectness.val:6.3f},{loss_objectness.avg:6.3f})  "
                        "loss_rpn_box_reg:({loss_rpn_box_reg.val:6.3f},{loss_rpn_box_reg.avg:6.3f})".format(
                        epoch=self.epoch,iter=i+1,num_iter=iter_nums,
                        loss=loss_all,
                        loss_cls=loss_classifier,
                        loss_box_reg=loss_box_reg,
                        loss_objectness=loss_objectness,
                        loss_rpn_box_reg=loss_rpn_box_reg
                        )                     
                    )
            self.train_epoch_loss_classifier.append(loss_classifier.avg)
            self.train_epoch_loss_box_reg.append(loss_box_reg.avg)
            self.train_epoch_loss_objectness.append(loss_objectness.avg)
            self.train_epoch_loss_rpn_box_reg.append(loss_rpn_box_reg.avg)
            self.train_epoch_loss_sum.append(loss_all.avg)   
            
            self.validate()
            
            self.save_checkpoint(True)
        
        
    def validate(self):
        self.net.eval()
        
        loss_all = AverageMeter()
        loss_classifier = AverageMeter()
        loss_box_reg = AverageMeter()
        loss_objectness = AverageMeter()
        loss_rpn_box_reg = AverageMeter()
        
        iter_nums = len(self.valloader)
        
        cpu_device = torch.device("cpu")
        
        coco = get_coco_api_from_dataset(self.valloader.dataset)
        iou_types =["bbox"]
        coco_evaluator = CocoEvaluator(coco, iou_types)
        
        for i,(inputs,targets) in enumerate(self.valloader):
            
            inputs = list(img.cuda() for img in inputs)
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            outputs = self.net(inputs)
                        
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)
            
            
            if i%self.print_freq == (self.print_freq-1):
                    self.logger.info(
                        "Test[{epoch:2d},{iter:5d}/{num_iter:d}] ".format(
                        epoch=self.epoch,iter=i+1,num_iter=iter_nums
                        
                        )                     
                    )
        
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
        # record AP result
        self.AP_50_95.append(coco_evaluator.coco_eval["bbox"].stats[0])
        self.AP_50.append(coco_evaluator.coco_eval["bbox"].stats[1])
        self.AP_75.append(coco_evaluator.coco_eval["bbox"].stats[2])
        self.AP_small.append(coco_evaluator.coco_eval["bbox"].stats[3])
        self.AP_medium.append(coco_evaluator.coco_eval["bbox"].stats[4])
        self.AP_large.append(coco_evaluator.coco_eval["bbox"].stats[5])
        self.AR_1.append(coco_evaluator.coco_eval["bbox"].stats[6])
        self.AR_10.append(coco_evaluator.coco_eval["bbox"].stats[7])
        self.AR_100.append(coco_evaluator.coco_eval["bbox"].stats[8])
        self.AR_small.append(coco_evaluator.coco_eval["bbox"].stats[9])
        self.AR_medium.append(coco_evaluator.coco_eval["bbox"].stats[10])
        self.AR_large.append(coco_evaluator.coco_eval["bbox"].stats[11])
        
        
            
        
    def accuracy(self,output,target,topk=(1,)):    
        maxk = max(topk)
        batch_size = target.size(0)
        
        values,pred_idx = output.topk(maxk,1,True,True)
        pred_idx = pred_idx.t()
        correct = pred_idx.eq(target.view(1,-1).expand_as(pred_idx))
        
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0/batch_size))
        return res
        
    def save_checkpoint(self,is_best):
        filename=os.path.join("./models",self.save_name+"_"+str(self.epoch)+".pth")
        torch.save({
            "epoch":self.epoch,
            "state_dict":self.net.state_dict(),
           
            "train_loss_classifier":self.train_loss_classifier,
            "train_loss_box_reg":self.train_loss_box_reg,
            "train_loss_objectness":self.train_loss_objectness,
            "train_loss_rpn_box_reg":self.train_loss_rpn_box_reg,
            "train_loss_sum":self.train_loss_sum,


            # loss per epoch
            "train_epoch_loss_classifier":self.train_epoch_loss_classifier,
            "train_epoch_loss_box_reg":self.train_epoch_loss_box_reg,
            "train_epoch_loss_objectness":self.train_epoch_loss_objectness,
            "train_epoch_loss_rpn_box_reg":self.train_epoch_loss_rpn_box_reg,
            "train_epoch_loss_sum":self.train_epoch_loss_sum,

            # coco val
            "AP_50_95":self.AP_50_95,
            "AP_50":self.AP_50,
            "AP_75":self.AP_75,
            "AP_small":self.AP_small,
            "AP_medium":self.AP_medium,
            "AP_large":self.AP_large,
            "AR_1":self.AR_1,
            "AR_10":self.AR_10,
            "AR_100":self.AR_100 ,
            "AR_small":self.AR_small,
            "AR_medium":self.AR_medium,
            "AR_large":self.AR_large,
        },filename)
        
        












