
import logging
import os
import torch
import shutil
import math



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
        
        
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = criterion
        self.optimizer = optimizer
        
        # statistics data
        self.net = net
        if check_point:
            self.load_checkpoint(check_point)
        else:
            # reset
            self.epoch = 0  # base on 1,not 0
            self.acc_epoch_train_top1 = []
            self.acc_epoch_train_top5 = []
            self.acc_epoch_val_top1 = []
            self.acc_epoch_val_top5 = []
            # loss per itertation
            self.loss_history_train = []
            self.loss_history_val = []
            # loss per epoch
            self.loss_epoch_train = []
            self.loss_epoch_val = []
            
            self.best_top1 = 0
        
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.print_freq = print_freq
        self.save_name = save_name
        # log
        self.logger = logging.getLogger(logfile)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(logfile)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        torch.backends.cudnn.benchmark = True
        
        self.logger.info("Solver init done...")
    
    def load_checkpoint(self,check_point):
        data = torch.load(check_point)
        self.net.load_state_dict(data["state_dict"])
        self.epoch = data["epoch"]
        self.acc_epoch_train_top1 = data["acc_epoch_train_top1"]
        self.acc_epoch_train_top5 = data["acc_epoch_train_top5"]
        self.acc_epoch_val_top1 = data["acc_epoch_val_top1"]
        self.acc_epoch_val_top5 = data["acc_epoch_val_top5"]
        # loss per itertation
        self.loss_history_train = data["loss_history_train"]
        self.loss_history_val = data["loss_history_val"]
        # loss per epoch
        self.loss_epoch_train = data["loss_epoch_train"]
        self.loss_epoch_val = data["loss_epoch_val"]
        self.best_top1 = data["best_top1"]
    
    
    def train(self,epochs):
        self.logger.info("begin training from epoch="+str(self.epoch+1))
        iter_nums = int(math.ceil(len(self.trainloader.dataset)/self.trainloader.batch_size))
        self.logger.info("There are "+ str(iter_nums) +" iterations per epoch.")
        for epoch in range(epochs):
            self.net.train()
            self.epoch = self.epoch + 1
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()          
            
            
            
            for i,(inputs,targets) in enumerate(self.trainloader):
                
                targets = targets - 1 #因为我存储的target是1 <= i <= classes_num
                inputs = inputs.cuda()
                targets = targets.cuda()
                
                outputs = self.net(inputs)
                loss = self.criterion(outputs,targets)
             
                # measure and record
                prec1,prec5 = self.accuracy(outputs,targets,topk=(1,5))
                losses.update(loss.item(),inputs.size(0))
                top1.update(prec1.item(),inputs.size(0))
                top5.update(prec5.item(),inputs.size(0))
                self.loss_history_train.append(loss.item())
                
                
                
                # BP
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                
                if i%self.print_freq == (self.print_freq-1):
                    self.logger.info(
                        "[{epoch:2d},{iter:5d}/{num_iter:d}] "
                        "loss:({loss.val:.5f},{loss.avg:.5f})  "
                        "Prec@1:({top1.val:6.3f},{top1.avg:6.3f})  "
                        "Prec@5:({top5.val:6.3f},{top5.avg:6.3f})".format(
                        epoch=self.epoch,iter=i+1,num_iter=iter_nums,
                        loss=losses,top1=top1,top5=top5    
                        )                     
                    )
            self.loss_epoch_train.append(losses.avg)
            self.acc_epoch_train_top1.append(top1.avg)
            self.acc_epoch_train_top5.append(top5.avg)
            
            self.validate()
            
            is_best = self.acc_epoch_val_top1[-1] > self.best_top1
            self.save_checkpoint(is_best)
        
        
    def validate(self):
        self.net.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        iter_nums = int(math.ceil(len(self.valloader.dataset)/self.valloader.batch_size))

        for i,(inputs,targets) in enumerate(self.valloader):
            targets = targets - 1
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            outputs = self.net(inputs)
            loss = self.criterion(outputs,targets)
                                 
            acc1,acc5 = self.accuracy(outputs,targets,(1,5))
            losses.update(loss.item(),inputs.size(0))
            top1.update(acc1.item(),inputs.size(0))
            top5.update(acc5.item(),inputs.size(0))
            self.loss_history_val.append(loss.item())
            
            
            if i%self.print_freq == (self.print_freq-1):
                    self.logger.info(
                        "[{epoch:2d},{iter:5d}/{num_iter:d}] "
                        "loss:({loss.val:.5f},{loss.avg:.5f})  "
                        "Prec@1:({top1.val:6.3f},{top1.avg:6.3f})  "
                        "Prec@5:({top5.val:6.3f},{top5.avg:6.3f})".format(
                        epoch=self.epoch,iter=i+1,num_iter=iter_nums,
                        loss=losses,top1=top1,top5=top5    
                        )                     
                    )
        self.loss_epoch_val.append(losses.avg)
        self.acc_epoch_val_top1.append(top1.avg)
        self.acc_epoch_val_top5.append(top5.avg)
            
        
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
            "acc_epoch_train_top1":self.acc_epoch_train_top1,
            "acc_epoch_train_top5":self.acc_epoch_train_top5,
            "acc_epoch_val_top1":self.acc_epoch_val_top1,
            "acc_epoch_val_top5":self.acc_epoch_val_top5,
            "loss_history_train":self.loss_history_train,
            "loss_history_val":self.loss_history_val,
            "loss_epoch_train":self.loss_epoch_train,
            "loss_epoch_val":self.loss_epoch_val,
            "print_freq":self.print_freq,
            "best_top1":self.best_top1
        },filename)
        if is_best:
            shutil.copyfile(filename,os.path.join("./models","best"+"_"+str(self.epoch)+".pth"))
        
        












