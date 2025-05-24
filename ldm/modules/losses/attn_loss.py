from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def loss_attn(attns, masks, vectors):
    B = masks.shape[0]

    loss = 0

    for i in range(len(attns)):

        b, l, n = attns[i].shape
        attn = rearrange(attns[i], '(B head) (W H) N -> B N head W H', B=B, 
                head=b//B, W=int(np.sqrt(l)), H=int(np.sqrt(l)))
        mask = F.interpolate(masks, size=(int(np.sqrt(l)), int(np.sqrt(l))), mode = 'bilinear')

        mask = mask.unsqueeze(2).expand(-1, -1, b//B, -1, -1)
        if mask.dtype != torch.bool:
            mask = (mask > 0.5).to(torch.bool)
        
        vector = vectors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, b//B, int(np.sqrt(l)), int(np.sqrt(l)))
        if vector.dtype != torch.bool:
            vector = (vector > 0.5).to(torch.bool)

        mask = mask & vector
        un_mask = (~mask) & vector
        
        fg_loss = torch.mean((1 - attn) * mask)
        bg_loss = torch.mean(attn * un_mask)

        loss += 0.5 * fg_loss + 0.5 * bg_loss
    
    loss[torch.isnan(loss)] = 0

    return loss/len(attns) 

import torch

def compute_loss(cross_attention_maps, image_masks, n_mask):
    """
    计算基于交叉注意力图和掩码的损失值
    
    参数:
        cross_attention_maps (List[torch.Tensor]): 多层注意力图列表，
            每个元素形状为 [batch*heads, h*w, n]
        image_mask (torch.Tensor): 图像掩码，形状为 [batch, n, h, w]
        n_mask (torch.Tensor): 序列掩码，形状为 [batch, n]
    
    返回:
        torch.Tensor: 计算得到的损失值
    """
    total_loss = 0.0
    B = image_masks.size(0)
    
    for attn_map in cross_attention_maps:
        B_times_H, hw, n = attn_map.shape
        H = B_times_H // B
        h = w = int(np.sqrt(hw))
        
        image_mask = F.interpolate(image_masks, size=(h, w), mode = 'bilinear')
        if image_mask.dtype != torch.bool:
            image_mask = (image_mask > 0.5).to(torch.bool)

        img_mask = image_mask.view(B, n, h, w)\
            .reshape(B, n, h*w)\
            .transpose(1, 2)\
            .unsqueeze(1)\
            .expand(-1, H, -1, -1)\
            .reshape(B*H, hw, n)
        
        # 调整n_mask形状 [B*H, hw, n]
        if n_mask.dtype != torch.bool:
            n_mask = (n_mask > 0.5).to(torch.bool)
        seq_mask = n_mask.unsqueeze(1)\
            .expand(-1, H, -1)\
            .reshape(B*H, n)\
            .unsqueeze(1)\
            .expand(-1, hw, -1)
        
        # 计算组合掩码
        valid_mask = img_mask & seq_mask
        invalid_mask = ~img_mask & seq_mask

        tensor_reshaped = valid_mask.view(B, H, h, w, n)
        tensor_reshaped = tensor_reshaped.permute(0, 4, 2, 1, 3).contiguous()
        tensor_reshaped = tensor_reshaped.view(h * B * n, w * H)
        image_array = tensor_reshaped.detach().cpu().numpy() * 255  # 归一化到 [0, 255]
        image_array = image_array.astype(np.uint8)
        img = Image.fromarray(image_array)
        img.save('valid_mask.png')  # 保存拼接后的图像

        tensor_reshaped = invalid_mask.view(B, H, h, w, n)
        tensor_reshaped = tensor_reshaped.permute(0, 4, 2, 1, 3).contiguous()
        tensor_reshaped = tensor_reshaped.view(h * B * n, w * H)
        image_array = tensor_reshaped.detach().cpu().numpy() * 255  # 归一化到 [0, 255]
        image_array = image_array.astype(np.uint8)
        img = Image.fromarray(image_array)
        img.save('invalid_mask.png')  # 保存拼接后的图像

        # tensor_reshaped = attn_map.view(B, H, h, w, n)
        
        tensor_reshaped = rearrange(attn_map, '(B head) (W H) N -> B head W H N', B=B, head=H, 
                              W=w, H=h)
        tensor_reshaped = tensor_reshaped[:,:,:,:,:2]
        tensor_reshaped = (tensor_reshaped - tensor_reshaped.min()) / (tensor_reshaped.max()-tensor_reshaped.min())
        tensor_reshaped = tensor_reshaped.permute(0, 4, 2, 1, 3).contiguous()
        tensor_reshaped = tensor_reshaped.view(h * B * 2, w * H)
        image_array = tensor_reshaped.detach().cpu().numpy() * 255  # 归一化到 [0, 255]
        image_array = image_array.astype(np.uint8)
        img = Image.fromarray(image_array)
        img.save('attn_map.png')  # 保存拼接后的图像
        
        # 计算各项损失
        term1 = safe_sum(attn_map, valid_mask, seq_mask)
        # term2 = safe_sum(attn_map, invalid_mask)
        
        total_loss += term1

    return total_loss / len(cross_attention_maps) if cross_attention_maps else torch.tensor(0.0)

def safe_sum(tensor, mask, seq_mask):
    """安全计算掩码区域均值"""
    loss = 1 - tensor[mask].sum() / (tensor[seq_mask].sum() + 1e-10)
    return loss
    

if __name__ == '__main__':
    attn_1 = torch.rand((16, 25, 3)).cuda()
    attn_2 = torch.rand((16, 9, 3)).cuda()
    attns = [attn_1, attn_2]
    masks = torch.rand((2, 3, 5, 5)).cuda()
    vectors = torch.tensor([[1,0,0],[1,0,0]]).cuda()
    loss = compute_loss(attns, masks, vectors)
    print(loss)