import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import AutoModel
import math
import copy
from typing import Optional
from timm.models.swin_transformer_v2 import swinv2_tiny_window16_256
import os
class Net(nn.Module):

    def __init__(self, current_path='./'):
        super(Net, self).__init__()

        dim_modelt = 2112
        self.L2pooling_l1 = L2pooling(channels=192)
        self.L2pooling_l2 = L2pooling(channels=384)
        self.L2pooling_l3 = L2pooling(channels=768)
        self.L2pooling_l4 = L2pooling(channels=768)
        self.model = swinv2_tiny_window16_256(pretrained=False)
        self.dim_modelt = dim_modelt

        nheadt = 16
        num_encoder_layerst = 2
        dim_feedforwardt = 64
        ddropout = 0.5
        normalize = False

        self.transformer = Transformer(
            d_model=dim_modelt,
            nhead=nheadt,
            num_encoder_layers=num_encoder_layerst,
            dim_feedforward=dim_feedforwardt,
            normalize_before=normalize,
            dropout=ddropout,
        )

        self.position_embedding = PositionEmbeddingSine(dim_modelt // 2,
                                                        normalize=True)

        self.fc2 = nn.Linear(dim_modelt, self.model.head.in_features)
        self.fc = nn.Linear(self.model.head.in_features * 2, 1)
        self.ScoreModel = ScoreModel()

        self.avg8 = nn.AvgPool2d((8, 8))
        self.avg4 = nn.AvgPool2d((4, 4))
        self.avg2 = nn.AvgPool2d((2, 2))

        self.drop2d = nn.Dropout(p=0.1)
        model_pretrained_name_or_path = os.path.join(current_path,'pretrained/PickScore_v1')
        self.pickmodel = AutoModel.from_pretrained(model_pretrained_name_or_path)

    def swin_forward(self, img):
        self.pos_enc_1 = self.position_embedding(
            torch.ones(1, self.dim_modelt, 8, 8).to(img.device))
        self.pos_enc = self.pos_enc_1.repeat(img.shape[0], 1, 1,
                                             1).contiguous()

        b,c,w,h=img.shape                   
        x=self.model.patch_embed(img)        
        x=self.model.pos_drop(x)
        x = self.model.layers[0](x)          
        layer1 = x.view(b, c*8*8, w//8, h//8)                       
        x = self.model.layers[1](x)                
        layer2 = x.view(b, c*8*8*2, w//16, h//16)                         
        x = self.model.layers[2](x)                 
        layer3 = x.view(b, c*8*8*4, w//32, h//32)                              
        x = self.model.layers[3](x)              
        layer4 = x.view(b, c*8*8*4, w//32, h//32)  
        layer1_t = self.avg4(
            self.drop2d(self.L2pooling_l1(F.normalize(layer1, dim=1, p=2))))
        layer2_t = self.avg2(
            self.drop2d(self.L2pooling_l2(F.normalize(layer2, dim=1, p=2))))
        layer3_t = self.drop2d(
            self.L2pooling_l3(F.normalize(layer3, dim=1, p=2)))
        layer4_t = self.drop2d(
            self.L2pooling_l4(F.normalize(layer4, dim=1, p=2)))
        layers = torch.cat((layer1_t, layer2_t, layer3_t, layer4_t), dim=1)

        out_t_c = self.transformer(layers, self.pos_enc)
        out_t_o = torch.flatten(self.avg8(out_t_c), start_dim=1)
        out_t_o = self.fc2(out_t_o)
        layer4_o = self.avg8(layer4)
        layer4_o = torch.flatten(layer4_o, start_dim=1)
        predictionQA = self.fc(
            torch.flatten(torch.cat((out_t_o, layer4_o), dim=1), start_dim=1))

        return predictionQA

    def pick_forward(self, image_inputs, text_inputs):
        image_embs = self.pickmodel.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = self.pickmodel.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = self.pickmodel.logit_scale.exp() * torch.bmm(text_embs.unsqueeze(1), image_embs.unsqueeze(2)).squeeze()

        return scores

    def forward(self, img, image_inputs, text_inputs):
        swin_score = self.swin_forward(img)
        pick_score = self.pick_forward(image_inputs, text_inputs)
        score = self.ScoreModel(torch.stack((swin_score.squeeze(), pick_score), dim=1))
        return score.squeeze(-1)

class Transformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        self._reset_parameters()


        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src , pos_embed):
        bs, c, h, w = src.shape
        src2 = src
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed2 = pos_embed

        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        
        memory = self.encoder(src, pos=pos_embed)
        
        return  memory.permute(1, 2, 0).view(bs, c, h, w)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output




class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats 
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_val):
        x = tensor_val
        mask = torch.gt(torch.zeros(x.shape),0).to( x.device)[:,0,:,:]
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=1, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer(
            "filter", g[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

    def forward(self, input):
        input = input**2
        out = F.conv2d(
            input,
            self.filter,
            stride=self.stride,
            padding=self.padding,
            groups=input.shape[1],
        )
        return (out + 1e-12).sqrt()


class ScoreModel(nn.Module):

    def __init__(self):
        super(ScoreModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, scores):
        x = F.relu(self.fc1(scores))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")