import torch
import torch.nn as nn
from ldm.modules.diffusionmodules.util import FourierEmbedder
from einops import repeat

class ObjNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 

        self.linears = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_text_feature = torch.nn.Parameter(torch.zeros([self.in_dim + self.position_dim]))

    def forward(self, obj_boxes, obj_attrs, obj_vectors, **kwargs):
        masks_ = obj_vectors.unsqueeze(-1)
        text_null = self.null_text_feature.view(1, 1, -1)

        xyxy_embedding = self.fourier_embedder(obj_boxes)
        text_embeddings = torch.cat([obj_attrs, xyxy_embedding], dim=-1)
        text_embeddings = text_embeddings * masks_ + (1 - masks_) * text_null

        text_feat = self.linears(text_embeddings)
   
        return text_feat
    
class WatNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs = fourier_freqs)
        self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 

        self.linears = nn.Sequential(
            nn.Linear( self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
  
    def forward(self, wat_boxes, wat_attrs, **kwargs):    
        _, N, _ = wat_attrs.shape 

        wat_boxes = repeat(
            wat_boxes, 
            'b n l -> b (n N) l',
            N = N)
        
        xyxy_embedding = self.fourier_embedder(wat_boxes) # B*N*4 --> B*N*C

        text_feat = self.linears(
            torch.cat([
                wat_attrs, 
                xyxy_embedding], 
                dim=-1)
        )

        return text_feat

class PositionNet(nn.Module):
    def __init__(self,  in_dim, out_dim):
        super().__init__()

        self.wat_processer = WatNet(in_dim, out_dim)
        self.obj_processer = ObjNet(in_dim, out_dim)
  
    def forward(self, wat_cond, obj_cond):

        wat_hints = self.wat_processer(**wat_cond)
        obj_hints = self.obj_processer(**obj_cond)

        return wat_hints, obj_hints


if __name__ == '__main__':
    wat_condition = dict(
        wat_attrs=torch.zeros(2, 77, 768).cuda(),
        wat_boxes=torch.zeros(2, 1, 4).cuda())
        
    obj_condition = dict(
        obj_masks=torch.zeros(2, 30, 512, 512).cuda(),
        obj_boxes=torch.zeros(2, 30, 4).cuda(),
        obj_attrs=torch.zeros(2, 30, 768).cuda(),
        obj_vectors=torch.zeros(2, 30).cuda())
    
    net = PositionNet(768, 768).cuda()
    out = net(wat_condition, obj_condition)
    print(out[0].shape, out[1].shape)
    

