from SAMA import ImageDataset
import os
import os.path
import pandas as pd
import torch
from transformers import AutoProcessor




class DataLoader(object):

    def __init__(self, istrain=True,batchsize=4,folder_path ='path',info_path ='path',current_path='./'):

        self.istrain = istrain
        self.batch_size=batchsize
        self.path = folder_path
        self.infopath = info_path
        self.processor = AutoProcessor.from_pretrained(os.path.join(current_path,'pretrained/processor'))


    def get_dataloader(self):
        files,prompt,labels=AIGCtestFolder(self.path, self.infopath)
        dataset=ImageDataset(files=files,prompts=prompt,labels=labels,is_train=self.istrain,stype="sama")
        data_sampler = None
        DataLoader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, sampler=data_sampler, pin_memory=True, drop_last=False, collate_fn=self.collate_fn)
        return DataLoader
    
    def collate_fn(self,batch):
        data, image, prompt, labels = zip(*batch)
        data = torch.stack(data)

        image_inputs = self.processor(
            images=image,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        torch.cuda.empty_cache()
        return data, image_inputs, text_inputs, labels

def AIGCtestFolder(root,infopath):
    imgname = []
    prompt=[]
    excel_file = infopath
    df = pd.read_excel(excel_file)
    for f_index, row in df.iterrows():
        imgname.append(row['name'])
        prompt.append(row['prompt'])

    sample = []
    for item in imgname:
        sample.append(os.path.join(root, item))

    return sample, prompt, imgname