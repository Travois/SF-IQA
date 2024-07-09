import torch
import dataloader
from tqdm import tqdm
import numpy as np
from Net import Net
import os
import argparse
import random
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='/AIGCQA-30K-Image/test',help='your datasets path')
    parser.add_argument('--infopath', type=str, default='/AIGCQA-30K-Image/info_test.xlsx',help='your datasets infofile path')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size to test image patches in')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='turn on flag to use GPU')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpus to use')
    parser.add_argument('--save_dir', type=str, default='./output',help='your output path')
    parser.add_argument('--use_tqdm', action='store_true', default=False, help='use the tqdm to show Progress bar')

    config = parser.parse_args()
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)
    np.random.seed(2024)
    random.seed(2024)
    
    if(not os.path.exists(config.save_dir)):
        os.mkdir(config.save_dir)

    current_path = os.path.dirname(os.path.abspath(__file__))
    if config.use_gpu:
        device = torch.device('cuda',config.gpu_id)
    else:
        device = torch.device("cpu")

    model = Net(current_path=current_path)
    model.to(device)
    data=dataloader.DataLoader(istrain=False,batchsize=config.batch_size,folder_path =config.datasets,info_path=config.infopath, current_path=current_path)

    path=[]
    pred_scores = []
    pred_rankscores = []
    pred_rplccscores = []
    pred_plccscores = []

    # 从文件中加载权重字典
    weights_dict = torch.load(os.path.join(current_path,'pretrained/weights_dict.pt'), map_location=device)

    # 根据键值选择需要的权重
    weights_base = weights_dict['base']
    weights_rank = weights_dict['rank']
    weights_rplcc = weights_dict['rplcc']
    weights_plcc = weights_dict['plcc']

    model.load_state_dict(weights_base, strict=False)
    model.eval()
    DataLoader=data.get_dataloader()
    pbartest = enumerate(DataLoader)
    if config.use_tqdm:
        pbartest = tqdm(pbartest, total=len(DataLoader), leave=False)
    with torch.no_grad():
        for _,batch in pbartest:
            img = batch[0]
            image_inputs = batch[1]
            text_inputs = batch[2]
            name = batch[3]
            img = torch.as_tensor(img.to(device)).requires_grad_(False)
            image_inputs = image_inputs.to(device)
            text_inputs = text_inputs.to(device)

            pred = model(img, image_inputs, text_inputs)
            pred_scores = pred_scores + pred.squeeze().cpu().tolist()
            path = path + name
        
    pred_scores = np.array(pred_scores)

    model.load_state_dict(weights_rank, strict=True)
    model.eval()
    DataLoader=data.get_dataloader()
    pbartest = enumerate(DataLoader)
    if config.use_tqdm:
        pbartest = tqdm(pbartest, total=len(DataLoader), leave=False)
    with torch.no_grad():
        for _,batch in pbartest:
            img = batch[0]
            image_inputs = batch[1]
            text_inputs = batch[2]
            name = batch[3]
            img = torch.as_tensor(img.to(device)).requires_grad_(False)
            image_inputs = image_inputs.to(device)
            text_inputs = text_inputs.to(device)

            pred = model(img, image_inputs, text_inputs)
            pred_rankscores = pred_rankscores + pred.squeeze().cpu().tolist()
        
    pred_rankscores = np.array(pred_rankscores)

    model.load_state_dict(weights_rplcc, strict=True)
    model.eval()
    DataLoader=data.get_dataloader()
    pbartest = enumerate(DataLoader)
    if config.use_tqdm:
        pbartest = tqdm(pbartest, total=len(DataLoader), leave=False)
    with torch.no_grad():
        for _,batch in pbartest:
            img = batch[0]
            image_inputs = batch[1]
            text_inputs = batch[2]
            name = batch[3]
            img = torch.as_tensor(img.to(device)).requires_grad_(False)
            image_inputs = image_inputs.to(device)
            text_inputs = text_inputs.to(device)

            pred = model(img, image_inputs, text_inputs)
            pred_rplccscores = pred_rplccscores + pred.squeeze().cpu().tolist()
        
    pred_rplccscores = np.array(pred_rplccscores)

    model.load_state_dict(weights_plcc, strict=True)
    model.eval()
    DataLoader=data.get_dataloader()
    pbartest = enumerate(DataLoader)
    if config.use_tqdm:
        pbartest = tqdm(pbartest, total=len(DataLoader), leave=False)
    with torch.no_grad():
        for _,batch in pbartest:
            img = batch[0]
            image_inputs = batch[1]
            text_inputs = batch[2]
            name = batch[3]
            img = torch.as_tensor(img.to(device)).requires_grad_(False)
            image_inputs = image_inputs.to(device)
            text_inputs = text_inputs.to(device)

            pred = model(img, image_inputs, text_inputs)
            pred_plccscores = pred_plccscores + pred.squeeze().cpu().tolist()

    pred_plccscores = np.array(pred_plccscores)
    dataPath = os.path.join(config.save_dir,'output.txt')
    scores = (pred_scores + pred_rplccscores + pred_rankscores + pred_plccscores)/4.0

    # 将名字和分数写入一个 txt 文件
    with open(dataPath, "w") as f:
        for name, score in zip(path, scores):
            f.write(f"{name},{score}\n")

if __name__ == '__main__':
    main()