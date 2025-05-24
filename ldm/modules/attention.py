from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

# from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
from torch.utils import checkpoint

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential( nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def fill_inf_from_mask(self, sim, mask):
        if mask is not None:
            B,M = mask.shape
            mask = mask.unsqueeze(1).repeat(1,self.heads,1).reshape(B*self.heads,1,-1)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        return sim

    def forward(self, x, key, value, mask=None):

        q = self.to_q(x)     # B*N*(H*C)
        k = self.to_k(key)   # B*M*(H*C)
        v = self.to_v(value) # B*M*(H*C)
   
        B, N, HC = q.shape 
        _, M, _ = key.shape
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C
        v = v.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale # (B*H)*N*M
        self.fill_inf_from_mask(sim, mask)
        attn = sim.softmax(dim=-1) # (B*H)*N*M

        out = torch.einsum('b i j, b j d -> b i d', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)

class objCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_k_box = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_box = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, obj_txt, obj_mask=None, obj_vector=None):

        q = self.to_q(x)
        k_obj = self.to_k_box(obj_txt)
        v_obj = self.to_v_box(obj_txt)

        B, N, HC = q.shape 
        _, M, _ = obj_txt.shape
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k_obj = k_obj.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C
        v_obj = v_obj.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C
        
        sim_obj = einsum('b i d, b j d -> b i j', q, k_obj) * self.scale

        max_neg_value = -torch.finfo(sim_obj.dtype).max
        _, l, _ = sim_obj.shape
        obj_vector = repeat(obj_vector, 'b n -> (b h) l n', h=self.heads, l=l)
        sim_copy = sim_obj.clone()
        sim_copy.masked_fill_(~obj_vector, max_neg_value)
        sim_obj = sim_copy

        attn_obj = sim_obj.softmax(dim=-1)

        # dropout unmask region
        max_neg_value = 0
        obj_mask = repeat(obj_mask, 'b j n-> (b h) j n', h=self.heads)
        sim_copy = sim_obj.clone()
        sim_copy.masked_fill_(~obj_mask, max_neg_value)
        attn_obj = sim_copy

        out = einsum('b i j, b j d -> b i d', attn_obj, v_obj)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)

        return self.to_out(out)
    

class WatCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_k_box = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_box = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, wat_txt, wat_mask=None, wat_vectors=None):

        q = self.to_q(x)
        k_obj = self.to_k_box(wat_txt)
        v_obj = self.to_v_box(wat_txt)

        B, N, HC = q.shape 
        _, M, _ = wat_txt.shape
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k_obj = k_obj.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C
        v_obj = v_obj.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H)*M*C
        
        sim_obj = einsum('b i d, b j d -> b i j', q, k_obj) * self.scale

        attn_obj = sim_obj.softmax(dim=-1)

        # dropout unmask region
        max_neg_value = 0
        wat_mask = repeat(wat_mask, 'b j n-> (b h) j n', h=self.heads)
        sim_copy = sim_obj.clone()
        sim_copy.masked_fill_(~wat_mask, max_neg_value)
        attn_obj = sim_copy

        out = einsum('b i j, b j d -> b i d', attn_obj, v_obj)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)

        return self.to_out(out) # * wat_vectors.unsqueeze(-1)

class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def forward(self, x):
        q = self.to_q(x) # B*N*(H*C)
        k = self.to_k(x) # B*N*(H*C)
        v = self.to_v(x) # B*N*(H*C)

        B, N, HC = q.shape 
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        v = v.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        attn = sim.softmax(dim=-1) # (B*H)*N*N

        out = torch.einsum('b i j, b j c -> b i c', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)



class GatedCrossAttentionDense(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head):
        super().__init__()
        
        self.attn = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head) 
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  

    def forward(self, x, obj):

        x_ca = self.attn( self.norm1(x), obj, obj) 
        x = x + self.scale*torch.tanh(self.alpha_attn) *  x_ca
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) ) 
        
        return x
    
class ObjCrossAttentionDense(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head):
        super().__init__()
        
        self.attn_obj = CrossAttention(query_dim=query_dim, 
                                       key_dim=key_dim, 
                                       value_dim=value_dim, 
                                       heads=n_heads, 
                                       dim_head=d_head) 
        self.attn_wat = CrossAttention(query_dim=query_dim, 
                                       key_dim=key_dim, 
                                       value_dim=value_dim, 
                                       heads=n_heads, 
                                       dim_head=d_head)
        self.attn_obj2wat = CrossAttention(query_dim=query_dim, 
                                           key_dim=query_dim, 
                                           value_dim=query_dim, 
                                           heads=n_heads, 
                                           dim_head=d_head)
        self.attn_wat2obj = CrossAttention(query_dim=query_dim, 
                                           key_dim=query_dim, 
                                           value_dim=query_dim, 
                                           heads=n_heads, 
                                           dim_head=d_head)

        self.ff = FeedForward(query_dim, query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.norm4 = nn.LayerNorm(query_dim)
        self.norm5 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn_obj', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_attn_wat', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        self.register_parameter('null_obj_feat', nn.Parameter(torch.zeros([query_dim, ])))
        self.register_parameter('null_wat_feat', nn.Parameter(torch.zeros([query_dim, ])))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  

    def forward(self, x, obj_hints=None, obj_masks=None, obj_vectors=None, wat_hints=None, wat_masks=None, wat_vectors=None):

        x_residual = x
        B, L, dim = x.shape
        x_obj_null = repeat(self.null_obj_feat, 'd -> b l d', b=B, l=L)
        x_wat_null = repeat(self.null_wat_feat, 'd -> b l d', b=B, l=L)
        obj_masks = repeat(obj_masks, 'b l n -> b n l c', c=dim)
        wat_masks = repeat(wat_masks, 'b l n -> b n l c', c=dim)
        num_objs = obj_masks.shape[1]

        x_flat = repeat(x, 'b l d -> b n l d', n=num_objs)
        x_flat = rearrange(x_flat, 'b n l d -> (b n) l d')
        obj_hints = rearrange(obj_hints, 'b n d -> (b n) d').unsqueeze(1)
        obj_vectors = rearrange(obj_vectors, 'b n -> (b n)').unsqueeze(-1)
        
        x_obj = self.attn_obj( self.norm1( x_flat ), obj_hints, obj_hints, mask=obj_vectors )
        x_obj = rearrange(x_obj, '(b n) l d -> b n l d', b=B, n=num_objs)
        x_obj = (x_obj * obj_masks).sum(dim=1) + ( 1 - obj_masks.max(dim=1).values ) * x_obj_null

        x_wat = self.attn_wat( self.norm2( x ), wat_hints, wat_hints )
        x_wat = (x_wat * wat_masks).sum(dim=1) + ( 1 - wat_masks.max(dim=1).values ) * x_wat_null

        x_wat_en = self.attn_obj2wat(self.norm3(x_wat), x_obj, x_obj)
        x_obj_en = self.attn_wat2obj(self.norm4(x_obj), x_wat, x_wat)

        x_obj_out = self.scale*torch.tanh( self.alpha_attn_obj ) * x_obj_en
        x_wat_out = self.scale*torch.tanh( self.alpha_attn_wat ) * x_wat_en * wat_vectors.unsqueeze(-1)
        x_out = x_residual + x_obj_out + x_wat_out
        
        x = x_out + self.scale * torch.tanh( self.alpha_dense ) * self.ff( self.norm5(x_out) ) 

        return x


class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim,  n_heads, d_head):
        super().__init__()
        
        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  


    def forward(self, x, obj):

        N_visual = x.shape[1]
        obj = self.linear(obj)

        x_sa = self.attn(  self.norm1(torch.cat([x,obj],dim=1))  )

        x = x + self.scale*torch.tanh(self.alpha_attn) * x_sa[:,0:N_visual,:]
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) )  
        
        return x


class GatedSelfAttentionDense2(nn.Module):
    def __init__(self, query_dim, context_dim,  n_heads, d_head):
        super().__init__()
        
        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  


    def forward(self, x, obj):

        B, N_visual, _ = x.shape
        B, N_ground, _ = obj.shape

        obj = self.linear(obj)
        
        # sanity check 
        size_v = math.sqrt(N_visual)
        size_g = math.sqrt(N_ground)
        assert int(size_v) == size_v, "Visual tokens must be square rootable"
        assert int(size_g) == size_g, "Grounding tokens must be square rootable"
        size_v = int(size_v)
        size_g = int(size_g)

        # select grounding token and resize it to visual token size as residual 
        out = self.attn(  self.norm1(torch.cat([x,obj],dim=1))  )[:,N_visual:,:]
        out = out.permute(0,2,1).reshape( B,-1,size_g,size_g )
        out = torch.nn.functional.interpolate(out, (size_v,size_v), mode='bicubic')
        residual = out.reshape(B,-1,N_visual).permute(0,2,1)
        
        # add residual to visual feature 
        x = x + self.scale*torch.tanh(self.alpha_attn) * residual
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) )  
        
        return x

class BasicTransformerBlock(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=True):
        super().__init__()
        self.attn1 = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)  
        self.ff = FeedForward(query_dim, glu=True)
        self.attn2 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head)  
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.use_checkpoint = use_checkpoint

        self.fuser_type = fuser_type

        if fuser_type == "gatedSA":
            # note key_dim here actually is context_dim
            self.fuser = GatedSelfAttentionDense(query_dim, key_dim, n_heads, d_head) 
        elif fuser_type == "gatedSA2":
            # note key_dim here actually is context_dim
            self.fuser = GatedSelfAttentionDense2(query_dim, key_dim, n_heads, d_head) 
        elif fuser_type == "gatedCA":
            self.fuser = GatedCrossAttentionDense(query_dim, key_dim, value_dim, n_heads, d_head) 
        elif fuser_type == "objCA":
            self.fuser = ObjCrossAttentionDense(query_dim, key_dim, value_dim, n_heads, d_head) 
        else:
            assert False 

    def forward(self, x, context, obj_hints=None, obj_masks=None, obj_vectors=None, wat_hints=None, wat_masks=None, wat_vectors=None):
#        return checkpoint(self._forward, (x, context, obj), self.parameters(), self.use_checkpoint)
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, context, obj_hints, obj_masks, obj_vectors, wat_hints, wat_masks, wat_vectors)
        else:
            return self._forward(x, context, obj_hints, obj_masks, obj_vectors, wat_hints, wat_masks, wat_vectors)

    def _forward(self, x, context, obj_hints=None, obj_masks=None, obj_vectors=None, wat_hints=None, wat_masks=None, wat_vectors=None): 
        
        x = self.attn1( self.norm1(x) ) + x
        if self.fuser_type == "objCA":
            x = self.fuser(x, obj_hints, obj_masks, obj_vectors, wat_hints, wat_masks, wat_vectors) # identity mapping in the beginning 
        else:
            x = self.fuser(x, obj_hints) # identity mapping in the beginning 
        x = self.attn2(self.norm2(x), context, context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, key_dim, value_dim, n_heads, d_head, depth=1, fuser_type=None, use_checkpoint=True):
        super().__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        query_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        
        self.proj_in = nn.Conv2d(in_channels,
                                 query_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=use_checkpoint)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(query_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
    def check_mask(self, mask):
        if mask.dtype != torch.bool:
            mask = (mask > 0.5).to(torch.bool)
        return mask
    def forward(self, x, context, obj_hints=None, obj_masks=None, obj_vectors=None, wat_hints=None, wat_masks=None, wat_vectors=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        # check mask
        obj_masks = self.check_mask(rearrange(obj_masks, 'b n h w -> b (h w) n')).to(torch.int)
        wat_masks = self.check_mask(rearrange(wat_masks, 'b n h w -> b (h w) n')).to(torch.int)
        obj_vectors = self.check_mask(obj_vectors)
        wat_vectors = wat_vectors

        for block in self.transformer_blocks:
            x = block(x, context, obj_hints, obj_masks, obj_vectors, wat_hints, wat_masks, wat_vectors)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        x = self.proj_out(x)
        return x + x_in

if __name__ == '__main__':
    model = SpatialTransformer(320, 768, 768, 8, 40, fuser_type='objCA').cuda()
    x = torch.rand((2, 320, 64, 64)).cuda()
    context = torch.rand((2, 77, 768)).cuda()
    obj_hints = torch.rand((2, 30, 768)).cuda()
    obj_masks = torch.rand((2, 30, 64, 64)).cuda()
    obj_vectors = torch.rand((2, 30)).cuda()
    weather_hint = torch.rand((2, 2, 16)).cuda()
    weather_vector = torch.rand((2, 2)).cuda()
    wat_hints = torch.rand((2, 77, 768)).cuda()
    wat_masks = torch.rand((2, 1, 64, 64)).cuda()
    wat_vectors = torch.rand((2, 1)).cuda()
    # h, emb, context, obj_hint, obj_masks, obj_vectors, wat_hint, wat_masks, wat_vectors
    out = model(x, context, obj_hints, obj_masks, obj_vectors, wat_hints, wat_masks, wat_vectors)
    print(out.shape)
