import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO



def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImgListDataset(torch.utils.data.Dataset):
    
    def __init__(self,root,transform,samples):
        self.root = root
        self.transform = transform
        self.samples = samples
        
        
    def __getitem__(self,index):
        path,target = self.samples[index]
        sample = pil_loader(os.path.join(self.root,path))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample,target
    
    def __len__(self):
        return len(self.samples)

class COCODataset(Dataset):
    
    def __init__(self,root,annotation,transforms):
        self.root = root
        self.transforms = transforms
        
        self.coco = COCO(annotation)
        ids = self.coco.getImgIds()
        imgs = []
        for img_id in ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids)>0:
                imgs.append(img_id)
        self.imgs = imgs
        self.cats = self.coco.getCatIds()
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,index):
        name = self.coco.loadImgs(self.imgs[index])[0]["file_name"]
        img = pil_loader(os.path.join(self.root,name))
        annIds = self.coco.getAnnIds(imgIds=[self.imgs[index]])
        anns = self.coco.loadAnns(annIds)
        
        # 这种是返回的target是一个字典，有两个属性，boxes：(N,4)，和labels:(N,)
        boxes = []
        labels = []
        for ann in anns:
            box = [0,0,0,0]
            box[0] = ann["bbox"][0]
            box[1] = ann["bbox"][1]
            box[2] = ann["bbox"][0] + ann["bbox"][2]
            box[3] = ann["bbox"][1] + ann["bbox"][3]
            boxes.append(box)
            labels.append(self.cats.index(ann["category_id"]))
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

    
#         # 这种返回的target是一个由dict组成的list，每个dict有两个属性，boxes:(4,) 和 labels: Int64
#         # 注意，loadAnns的结果本身是个list，所以这种方式其实是它的一个子集
#         target = []
#         for ann in anns:
#             item = {}
#             box = [0,0,0,0]
#             box[0] = ann["bbox"][0]
#             box[1] = ann["bbox"][1]
#             box[2] = ann["bbox"][0] + ann["bbox"][2]
#             box[3] = ann["bbox"][1] + ann["bbox"][3]
#             item["boxes"] = box
#             item["labels"] = self.cats.index(ann["category_id"])
#             target.append(item)
        if self.transforms is not None:
            img,target = self.transforms(img,target)
#         print("COCO:",index,'\tboxes.len:',len(target["boxes"]),"imgID:",self.imgs[index])
        return img,target



