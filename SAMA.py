import torch
from torchvision import transforms

from PIL import Image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, files, prompts, labels, 
                 data_args={"fwin_h": 8, "fwin_w": 8, "fsize_h": 32, "fsize_w": 32}, 
                 stype="fragment",
                 is_train=True):
        
        super().__init__()
        
        self.files = files 
        self.prompts=prompts
        self.labels = labels
        self.is_train = is_train
        self.length = len(files)

        self.fwin_h = data_args['fwin_h']
        self.fwin_w = data_args['fwin_w']
        self.fsize_h = data_args['fsize_h']
        self.fsize_w = data_args['fsize_w']

        self.minh = self.fwin_h * self.fsize_h
        self.minw = self.fwin_w * self.fsize_w
        self.minsize = max(self.minh, self.minw)

        self.stype = stype if stype in ["sama", "sama-spm"] else "fragment"

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(45),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    def get_spatial_fragments(self, img, fragments_h=8, fragments_w=8, fsize_h=32, fsize_w=32):
        size_h = fragments_h * fsize_h
        size_w = fragments_w * fsize_w

        res_h, res_w = img.shape[-2:]
        ratio = min(res_h / size_h, res_w / size_w)
        if ratio < 1:
            img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=1 / ratio, mode="bilinear", align_corners=False)
            img = img[0]
        size = size_h, size_w

        ## make sure that sampling will not run out of the picture
        hgrids = torch.LongTensor([min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)])
        wgrids = torch.LongTensor([min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)])
        hlength, wlength = res_h // fragments_h, res_w // fragments_w

        if self.is_train:
            if hlength > fsize_h:
                rnd_h = torch.randint(hlength - fsize_h, (len(hgrids), len(wgrids)))
            else:
                rnd_h = torch.zeros((len(hgrids), len(wgrids))).int()
            if wlength > fsize_w:
                rnd_w = torch.randint(wlength - fsize_w, (len(hgrids), len(wgrids)))
            else:
                rnd_w = torch.zeros((len(hgrids), len(wgrids))).int()
        else:
            rnd_h = torch.ones((len(hgrids), len(wgrids))).int() * int((hlength - fsize_h) / 2)
            rnd_w = torch.ones((len(hgrids), len(wgrids))).int() * int((wlength - fsize_w) / 2) 

        t_img = torch.zeros(img.shape[:-2] + size).to(img.device)

        for i, hs in enumerate(hgrids):
            for j, ws in enumerate(wgrids):
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w

                h_so, h_eo = hs + rnd_h[i][j], hs + rnd_h[i][j] + fsize_h
                w_so, w_eo = ws + rnd_w[i][j], ws + rnd_w[i][j] + fsize_w
                t_img[:, h_s:h_e, w_s:w_e] = img[:, h_so:h_eo, w_so:w_eo]
        return t_img


    def get_spatial_fragments_spm(self, img, fragments_h=8, fragments_w=8, fsize_h=32, fsize_w=32):
        size_h = fragments_h * fsize_h
        size_w = fragments_w * fsize_w

        res_h, res_w = img.shape[-2:]
        ratio = min(res_h / size_h, res_w / size_w)
        if ratio < 1:
            res_h, res_w = round(res_h / ratio), round(res_w / ratio)
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(res_h, res_w), mode="bilinear", align_corners=False)
            img = img[0]
            ratio = min(res_h / size_h, res_w / size_w)
        size = size_h, size_w

        img_scale, hgrids, wgrids = [], [], []
        rnd_h, rnd_w = [], []
        if self.is_train:
            rnd_rh, rnd_rw = torch.rand((fragments_h, fragments_w)), torch.rand((fragments_h, fragments_w))
        else:
            rnd_rh, rnd_rw = torch.ones((fragments_h, fragments_w)) * 0.5, torch.ones((fragments_h, fragments_w)) * 0.5

        factors = [1, 1 / ratio]
        for scale in factors:
            this_h, this_w = round(res_h * scale), round(res_w * scale)
            img_scale.append(torch.nn.functional.interpolate(img.unsqueeze(0), size=(this_h, this_w), mode="bilinear", align_corners=False)[0])

            hgrids.append(torch.LongTensor([min(this_h // fragments_h * i, this_h - fsize_h) for i in range(fragments_h)]))
            wgrids.append(torch.LongTensor([min(this_w // fragments_w * i, this_w - fsize_w) for i in range(fragments_w)]))

            hlength, wlength = this_h // fragments_h, this_w // fragments_w
            rnd_h.append((rnd_rh[:, :] * (hlength - fsize_h)).int())
            rnd_w.append((rnd_rw[:, :] * (wlength - fsize_w)).int())

        target_imgs = torch.zeros((2, ) + img.shape[:-2] + size).to(img.device)
        for k, scale in enumerate(factors):
            for i, hs in enumerate(hgrids[k]):
                for j, ws in enumerate(wgrids[k]):
                    h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                    w_s, w_e = j * fsize_w, (j + 1) * fsize_w

                    h_so = hs + rnd_h[k][i][j]
                    h_eo = h_so + fsize_h
                    w_so = ws + rnd_w[k][i][j]
                    w_eo = w_so + fsize_w
                    target_imgs[k, :, h_s:h_e, w_s:w_e] = img_scale[k][:, h_so:h_eo, w_so:w_eo]  # 32 * 32

        # patch-based mask [4, 4]
        mask = torch.zeros((1, size_h, size_w))
        for i in range(size_w // 8):  # patch为4
            for j in range(size_h // 8):
                mask[:, j*8:j*8+4, i*8:i*8+4] = 1
                mask[:, j*8+4:j*8+8, i*8+4:i*8+8] = 1

        out_img = mask * target_imgs[0] + (1 - mask) * target_imgs[1]
        return out_img

    def get_spatial_fragments_swm(self, img, fragments_h=8, fragments_w=8, fsize_h=32, fsize_w=32):
        size_h = fragments_h * fsize_h
        size_w = fragments_w * fsize_w

        res_h, res_w = img.shape[-2:]
        ratio = min(res_h / size_h, res_w / size_w)
        if ratio < 1:
            res_h, res_w = round(res_h / ratio), round(res_w / ratio)
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(res_h, res_w), mode="bilinear", align_corners=False)
            img = img[0]
            ratio = min(res_h / size_h, res_w / size_w)
        size = size_h, size_w

        img_scale, hgrids, wgrids = [], [], []
        rnd_h, rnd_w = [], []
        if self.is_train:
            rnd_rh, rnd_rw = torch.rand((fragments_h, fragments_w)), torch.rand((fragments_h, fragments_w))
        else:
            rnd_rh, rnd_rw = torch.ones((fragments_h, fragments_w)) * 0.5, torch.ones((fragments_h, fragments_w)) * 0.5

        factors = [1, 1 / ratio]
        for scale in factors:
            this_h, this_w = round(res_h * scale), round(res_w * scale)
            img_scale.append(torch.nn.functional.interpolate(img.unsqueeze(0), size=(this_h, this_w), mode="bilinear", align_corners=False)[0])

            hgrids.append(torch.LongTensor([min(this_h // fragments_h * i, this_h - fsize_h) for i in range(fragments_h)]))
            wgrids.append(torch.LongTensor([min(this_w // fragments_w * i, this_w - fsize_w) for i in range(fragments_w)]))

            hlength, wlength = this_h // fragments_h, this_w // fragments_w
            rnd_h.append((rnd_rh[:, :] * (hlength - fsize_h)).int())
            rnd_w.append((rnd_rw[:, :] * (wlength - fsize_w)).int())

        target_imgs = torch.zeros((2, ) + img.shape[:-2] + size).to(img.device)
        for k, scale in enumerate(factors):
            for i, hs in enumerate(hgrids[k]):
                for j, ws in enumerate(wgrids[k]):
                    h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                    w_s, w_e = j * fsize_w, (j + 1) * fsize_w

                    h_so = hs + rnd_h[k][i][j]
                    h_eo = h_so + fsize_h
                    w_so = ws + rnd_w[k][i][j]
                    w_eo = w_so + fsize_w
                    target_imgs[k, :, h_s:h_e, w_s:w_e] = img_scale[k][:, h_so:h_eo, w_so:w_eo]  # 32 * 32

        # window-based mask [32, 32]
        mask = torch.zeros((1, size_h, size_w))
        for i in range(fragments_h):  # window
            for j in range(fragments_w):
                if (i + j) % 2 == 0:
                    mask[:, j*32:j*32+32, i*32:i*32+32] = 1

        out_img = mask * target_imgs[0] + (1 - mask) * target_imgs[1]
        return out_img
    
    def __getitem__(self, index):
        filename = self.files[index]
        prompt = self.prompts[index]
        label = self.labels[index]
        
        image = Image.open(filename).convert('RGB')
        width, height = image.size
        img = image
        if min(width, height) < self.minsize:
            scale_factor = self.minsize / min(width, height)
            img = img.resize((int(width * scale_factor), int(height * scale_factor)), Image.BILINEAR)

        img = self.transform(img)

        if self.stype == "fragment":
            data = self.get_spatial_fragments(img, self.fwin_h, self.fwin_w, self.fsize_h, self.fsize_w)
        elif self.stype == "sama-spm":
            data = self.get_spatial_fragments_spm(img, self.fwin_h, self.fwin_w, self.fsize_h, self.fsize_w)
        elif self.stype == "sama":
            data = self.get_spatial_fragments_swm(img, self.fwin_h, self.fwin_w, self.fsize_h, self.fsize_w)
        else:
            raise NotImplementedError

        return data, image, prompt, label
    
    def __len__(self):
        return self.length


