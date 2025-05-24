import torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import numpy as np
import random
import time 
from ldm.data.base import collate_fn
from ldm.data.dataset import ShipGDData
from torch.utils.data.distributed import  DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os 
import shutil
import torchvision
from ldm.modules.convert_ckpt import add_additional_channels
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from ldm.modules.distributed import get_rank, synchronize, get_world_size
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from copy import deepcopy
# from inpaint_mask_func import draw_masks_from_boxes
from ldm.modules.attention import BasicTransformerBlock
import matplotlib.pyplot as plt
from ldm.modules.losses.attn_loss import loss_attn, compute_loss
import torch.nn.functional as F
try:
    from apex import amp
except:
    pass  
# = = = = = = = = = = = = = = = = = = useful functions = = = = = = = = = = = = = = = = = #

def save_heatmap(attns, vectors):
        
    attn_vis = torch.mean(attns[0], dim=2)
    b,n,h,w = attn_vis.shape
    for i in range(b):
        for j in range(n):
            if vectors[i, j] == 1:
                plt.figure(figsize=(64, 64))
                plt.imshow(attn_vis[i, j, :, :].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
                plt.title(f'Heatmap of batch{i}, {j}-th object.')
                plt.savefig(f'{i}_{j}.png')

class ImageCaptionSaver:
    def __init__(self, base_path, nrow=8, normalize=True, scale_each=True, range=(-1,1) ):
        self.base_path = base_path 
        self.nrow = nrow
        self.normalize = normalize
        self.scale_each = scale_each
        self.range = range

    def __call__(self, images, real, seen):

        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'.png')
        image_fin = []
        for i in range(len(images)):
            image_fin.append(torch.cat((real[i], images[i]), dim=2))
        final = torch.cat(image_fin, dim=1)
        torchvision.utils.save_image( final, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range )



def read_official_ckpt(ckpt_path):      
    "Read offical pretrained SD ckpt and convert into my style" 
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    out = {}
    out["model"] = {}
    out["text_encoder"] = {}
    out["autoencoder"] = {}
    out["unexpected"] = {}
    out["diffusion"] = {}

    for k,v in state_dict.items():
        if k.startswith('model.diffusion_model'):
            out["model"][k.replace("model.diffusion_model.", "")] = v 
        elif k.startswith('cond_stage_model'):
            out["text_encoder"][k.replace("cond_stage_model.", "")] = v 
        elif k.startswith('first_stage_model'):
            out["autoencoder"][k.replace("first_stage_model.", "")] = v 
        elif k in ["model_ema.decay", "model_ema.num_updates"]:
            out["unexpected"][k] = v  
        else:
            out["diffusion"][k] = v     
    return out 

def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def sub_batch(batch, num=1):
    # choose first num in given batch 
    num = num if num > 1 else 1 
    for k in batch:
        batch[k] = batch[k][0:num]
    return batch


def wrap_loader(loader):
    while True:
        for batch in loader:  # TODO: it seems each time you have the same order for all epoch?? 
            yield batch


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def count_params(params):
    total_trainable_params_count = 0 
    for p in params:
        total_trainable_params_count += p.numel()
    print("total_trainable_params_count is: ", total_trainable_params_count)


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

           
def create_expt_folder_with_auto_resuming(ckpt_out, name):
    name = os.path.join( ckpt_out, name )
    writer = None
    checkpoint = None

    if os.path.exists(name):
        all_tags = os.listdir(name)
        all_existing_tags = [ tag for tag in all_tags if tag.startswith('tag')    ]
        all_existing_tags.sort()
        all_existing_tags = all_existing_tags[::-1]
        for previous_tag in all_existing_tags:
            potential_ckpt = os.path.join( name, previous_tag, 'checkpoint_latest.pth' )
            if os.path.exists(potential_ckpt):
                checkpoint = potential_ckpt
                if get_rank() == 0:
                    print('auto-resuming ckpt found '+ potential_ckpt)
                break 
        curr_tag = 'tag'+str(len(all_existing_tags)).zfill(2)
        name = os.path.join( name, curr_tag ) # output/name/tagxx
    else:
        name = os.path.join( name, 'tag00' ) # output/name/tag00

    if get_rank() == 0:
        os.makedirs(name) 
        os.makedirs(  os.path.join(name,'Log')  ) 
        writer = SummaryWriter( os.path.join(name,'Log')  )

    return name, writer, checkpoint

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 

class Trainer:
    def __init__(self, config):

        self.config = config
        self.device = torch.device("cuda")

        self.l_simple_weight = 1
        self.l_fg_weight = 0.3
        self.water_colors = None 
        self.obj_colors = None
        self.name, self.writer, checkpoint = create_expt_folder_with_auto_resuming(config.ckpt_out, config.name)
        if get_rank() == 0:
            shutil.copyfile(config.yaml_file, os.path.join(self.name, "train_config_file.yaml")  )
            self.config_dict = vars(config)
            torch.save(  self.config_dict,  os.path.join(self.name, "config_dict.pth")     )


        # = = = = = = = = = = = = = = = = = create model and diffusion = = = = = = = = = = = = = = = = = #
        self.model = instantiate_from_config(config.model).to(self.device)
        self.autoencoder = instantiate_from_config(config.autoencoder).to(self.device)
        self.text_encoder = instantiate_from_config(config.text_encoder).to(self.device)
        self.diffusion = instantiate_from_config(config.diffusion).to(self.device)

        state_dict = read_official_ckpt(  config.official_ckpt_name   )
        
        # modify the input conv for SD if necessary (grounding as unet input; inpaint)
        additional_channels = self.model.additional_channel_from_downsampler
        if self.config.inpaint_mode:
            additional_channels += 5 # 5 = 4(latent) + 1(mask)
        add_additional_channels(state_dict["model"], additional_channels)
        self.input_conv_train = True if additional_channels>0 else False

        # load original SD ckpt (with inuput conv may be modified) 
        missing_keys, unexpected_keys = self.model.load_state_dict( state_dict["model"], strict=False  )
        assert unexpected_keys == []
        original_params_names = list( state_dict["model"].keys()  ) # used for sanity check later 
        
        self.autoencoder.load_state_dict( state_dict["autoencoder"]  )
        self.text_encoder.load_state_dict( state_dict["text_encoder"]  )
        missing_keys, unexpected_keys = self.diffusion.load_state_dict( state_dict["diffusion"], strict=False  )
        assert missing_keys == []
 
        self.autoencoder.eval()
        self.text_encoder.eval()
        disable_grads(self.autoencoder)
        disable_grads(self.text_encoder)

        # = = = = = = = = = = = = = load from ckpt: (usually for inpainting training) = = = = = = = = = = = = = #
        if self.config.ckpt is not None:
            first_stage_ckpt = torch.load(self.config.ckpt, map_location="cpu")
            self.model.load_state_dict(first_stage_ckpt["model"])

        # = = = = = = = = = = = = = = = = = create opt = = = = = = = = = = = = = = = = = #
        params = []
        trainable_names = []
        all_params_name = []
        for name, p in self.model.named_parameters():
            # if ("transformer_blocks" in name):
            if ("transformer_blocks" in name) and ("fuser" in name):
                # New added Attention layers 
                params.append(p) 
                trainable_names.append(name)
            elif  "position_net" in name:
                # Grounding token processing network 
                params.append(p) 
                trainable_names.append(name)
            else:

                assert name in original_params_names, name 
            all_params_name.append(name) 

        self.opt = torch.optim.AdamW(params, lr=config.base_learning_rate, weight_decay=config.weight_decay) 
        count_params(params)
        
        #  = = = = = EMA... It is worse than normal model in early experiments, thus never enabled later = = = = = = = = = #
        if config.enable_ema:
            self.master_params = list(self.model.parameters()) 
            self.ema = deepcopy(self.model)
            self.ema_params = list(self.ema.parameters())
            self.ema.eval()

        # = = = = = = = = = = = = = = = = = = = = create scheduler = = = = = = = = = = = = = = = = = = = = #
        if config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps, num_training_steps=config.total_iters)
        elif config.scheduler_type == "constant":
            self.scheduler = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps)
        else:
            assert False 

        # = = = = = = = = = = = = = = = = = = = = create data = = = = = = = = = = = = = = = = = = = = #  
        dataset_train = ShipGDData(config.data.base_path, resolution=config.data.resolution, mode=config.data.mode)
        sampler = DistributedSampler(dataset_train, seed=config.seed) if config.distributed else None 
        loader_train = DataLoader( dataset_train,  batch_size=config.batch_size, 
                                                   shuffle=(sampler is None),
                                                   num_workers=config.workers, 
                                                   pin_memory=True,
                                                   collate_fn=collate_fn, 
                                                   sampler=sampler)
        self.dataset_train = dataset_train
        self.loader_train = wrap_loader(loader_train)

        if get_rank() == 0:
            total_image = len(dataset_train)
            print("Total training images: ", total_image)     
        

        # = = = = = = = = = = = = = = = = = = = = load from autoresuming ckpt = = = = = = = = = = = = = = = = = = = = #
        self.starting_iter = 0  
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            if config.enable_ema:
                self.ema.load_state_dict(checkpoint["ema"])
            self.opt.load_state_dict(checkpoint["opt"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.starting_iter = checkpoint["iters"]
            if self.starting_iter >= config.total_iters:
                synchronize()
                print("Training finished. Start exiting")
                exit()

        # = = = = = = = = = = = = = = = = = = = = misc and ddp = = = = = = = = = = = = = = = = = = = =#    
        
        # func return input for grounding tokenizer 
        self.grounding_tokenizer_input = instantiate_from_config(config.grounding_tokenizer_input)
        self.model.grounding_tokenizer_input = self.grounding_tokenizer_input
        
        # func return input for grounding downsampler  
        self.grounding_downsampler_input = None
        if 'grounding_downsampler_input' in config:
            self.grounding_downsampler_input = instantiate_from_config(config.grounding_downsampler_input)

        if get_rank() == 0:       
            self.image_caption_saver = ImageCaptionSaver(self.name)

        if config.distributed:
            self.model = DDP( self.model, device_ids=[config.local_rank], output_device=config.local_rank, broadcast_buffers=False )

    @torch.no_grad()
    def get_input(self, batch):
        z = self.autoencoder.encode( batch["image"] )

        context = self.text_encoder.encode( batch["caption"]  )
        wat_attrs = self.text_encoder.encode( batch["wat_attrs"]  )

        _t = torch.rand(z.shape[0]).to(z.device)
        t = (torch.pow(_t, 1) * 1000).long()
        t = torch.where(t!=1000, t, 999) # if 1000, then replace it with 999
        
        inpainting_extra_input = None    
        
        grounding_extra_input = None
        if self.grounding_downsampler_input != None:
            grounding_extra_input = self.grounding_downsampler_input.prepare(batch)

        return z, t, context, wat_attrs, inpainting_extra_input, grounding_extra_input 
    def _upsample(self, x, y):
        _, _, H0, W0 = y.size()
        return F.interpolate(x, size=(H0, W0), mode = 'bilinear')

    def run_one_step(self, batch):
        x_start, t, context, wat_attrs, inpainting_extra_input, grounding_extra_input = self.get_input(batch)
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)

        grounding_input = self.grounding_tokenizer_input.prepare(batch, wat_attrs)
        input = dict(x=x_noisy, 
                    timesteps=t, 
                    context=context, 
                    inpainting_extra_input=inpainting_extra_input,
                    grounding_extra_input=grounding_extra_input,
                    grounding_input=grounding_input)
        model_output = self.model(input)
        
        loss = torch.nn.functional.mse_loss(model_output, noise) * self.l_simple_weight

        self.loss_dict = {"loss": loss.item()}

        return loss 
        
    def start_training(self):

        iterator = tqdm(range(self.starting_iter, self.config.total_iters), desc='Training progress',  disable=get_rank() != 0 )
        self.model.train()
        for iter_idx in iterator: # note: iter_idx is not from 0 if resume training
            self.iter_idx = iter_idx

            self.opt.zero_grad()
            batch = next(self.loader_train)
            batch_to_device(batch, self.device)

            loss = self.run_one_step(batch)
            loss.backward()
            self.opt.step() 
            self.scheduler.step()
            if self.config.enable_ema:
                update_ema(self.ema_params, self.master_params, self.config.ema_rate)


            if (get_rank() == 0):
                if (iter_idx % 10 == 0):
                    self.log_loss() 
                if (iter_idx == 0)  or  ( iter_idx % self.config.save_every_iters == 0 )  or  (iter_idx == self.config.total_iters-1):
                    self.save_ckpt_and_result()
            synchronize()

        synchronize()
        print("Training finished. Start exiting")
        exit()


    def log_loss(self):
        for k, v in self.loss_dict.items():
            self.writer.add_scalar(  k, v, self.iter_idx+1  )  # we add 1 as the actual name
    
    @torch.no_grad()
    def save_ckpt_and_result(self):

        model_wo_wrapper = self.model.module if self.config.distributed else self.model

        iter_name = self.iter_idx + 1     # we add 1 as the actual name

        if not self.config.disable_inference_in_training:
            # Do an inference on one training batch 
            batch_here = self.config.batch_size
            # batch = sub_batch( next(self.loader_train), batch_here)
            batch = next(self.loader_train)
            batch_to_device(batch, self.device)

            #if "boxes" in batch:
            real_images_with_box_drawing = [] # we save this durining trianing for better visualization
            for i in range(batch_here):
                inp = dict(
                    image = batch["image"][i].detach().cpu().numpy().copy(),
                    caption = batch["caption"][i],
                    wat_masks = batch["wat_masks"][i].detach().cpu().numpy().copy(),
                    wat_attrs = batch["wat_attrs"][i],
                    wat_vectors = batch["wat_vectors"][i].detach().cpu().numpy().copy(),
                    obj_boxes = batch["obj_boxes"][i].detach().cpu().numpy().copy(),
                    obj_classes = batch["obj_classes"][i].copy(),
                    obj_attrs = batch["obj_attrs"][i].copy(),
                    obj_vectors = batch["obj_vectors"][i].detach().cpu().numpy().copy(),
                    return_type = 'tensor')
                im = self.dataset_train._disply(**inp)
                real_images_with_box_drawing.append(im)
            real_images_with_box_drawing = torch.stack(real_images_with_box_drawing)
            
            uc = self.text_encoder.encode( batch_here*[""] )
            context = self.text_encoder.encode(  batch["caption"]  )
            wat_attrs = self.text_encoder.encode(  batch["wat_attrs"]  )
            
            plms_sampler = PLMSSampler(self.diffusion, model_wo_wrapper)      
            shape = (batch_here, model_wo_wrapper.in_channels, model_wo_wrapper.image_size, model_wo_wrapper.image_size)

            grounding_input = self.grounding_tokenizer_input.prepare(batch, wat_attrs)
            input = dict( x=None, 
                          timesteps=None, 
                          context=context, 
                          inpainting_extra_input=None,
                          grounding_extra_input=None,
                          grounding_input=grounding_input )
            samples = plms_sampler.sample(S=50, shape=shape, input=input, uc=uc, guidance_scale=5)
            
            autoencoder_wo_wrapper = self.autoencoder # Note itself is without wrapper since we do not train that. 
            samples = autoencoder_wo_wrapper.decode(samples).cpu()
            samples = torch.clamp(samples, min=-1, max=1)

            self.image_caption_saver(samples, real_images_with_box_drawing, iter_name)

        ckpt = dict(model = model_wo_wrapper.state_dict(),
                    text_encoder = self.text_encoder.state_dict(),
                    autoencoder = self.autoencoder.state_dict(),
                    diffusion = self.diffusion.state_dict(),
                    opt = self.opt.state_dict(),
                    scheduler= self.scheduler.state_dict(),
                    iters = self.iter_idx+1,
                    config_dict=self.config_dict,
        )
        if self.config.enable_ema:
            ckpt["ema"] = self.ema.state_dict()
        torch.save( ckpt, os.path.join(self.name, "checkpoint_"+str(iter_name).zfill(8)+".pth") )
        torch.save( ckpt, os.path.join(self.name, "checkpoint_latest.pth") )


