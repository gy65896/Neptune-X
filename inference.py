
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse, torch, os, cv2

from ldm.data.base import BaseData, draw_bounding_boxes
from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler

device = "cuda" if torch.cuda.is_available() else "cpu"

def batch_to_device(batch, device, force_float32=True):

    if isinstance(batch, (np.ndarray, np.generic)):
        batch = torch.from_numpy(batch)
        if force_float32 and batch.is_floating_point() and batch.dtype != torch.float32:
            batch = batch.to(torch.float32)
        batch = batch.to(device)
        return batch
    elif isinstance(batch, torch.Tensor):
        if force_float32 and batch.is_floating_point() and batch.dtype != torch.float32:
            batch = batch.to(torch.float32)
        batch = batch.to(device)
        return batch
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        batch = [batch_to_device(v, device) for v in batch]
        return batch
    else:
        return batch

class Processer:
    def __init__(self, 
            args,
            max_obj=30):

        self.batch_size = args.batch_size
        self.guidance_scale = args.guidance_scale
        self.negative_prompt = args.negative_prompt
        self.max_obj = max_obj
        
        os.makedirs(args.out_path, exist_ok=True)
        self.image_dir = os.path.join(args.out_path,'image')
        os.makedirs(self.image_dir, exist_ok=True)
        self.image_w_box_dir = os.path.join(args.out_path,'img_w_box')
        os.makedirs(self.image_w_box_dir, exist_ok=True)

        self.load_sd_ckpt(args.sd_ckpt, args.steps)

    def load_sd_ckpt(self, ckpt_path, steps):
    
        saved_ckpt = torch.load(ckpt_path)
        config = saved_ckpt["config_dict"]["_content"]

        model = instantiate_from_config(config['model']).to(device).eval()
        autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
        text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
        diffusion = instantiate_from_config(config['diffusion']).to(device)

        model.load_state_dict( saved_ckpt['model'] )
        autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
        text_encoder.load_state_dict( saved_ckpt["text_encoder"]  ,strict=False)
        diffusion.load_state_dict( saved_ckpt["diffusion"]  )

        grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
        model.grounding_tokenizer_input = grounding_tokenizer_input

        self.sampler = PLMSSampler(diffusion, model)
        self.steps = steps 
        self.text_encoder = text_encoder
        self.uc = self.text_encoder.encode( self.batch_size*[args.negative_prompt] )
        self.grounding_tokenizer_input = grounding_tokenizer_input
        self.model = model
        self.autoencoder = autoencoder
    
    @torch.no_grad()
    def generator(self, cond):
        caption = cond['caption']
        boxes = cond['boxes']
        classes = cond['classes']
        water_boxes = cond['water_boxes']
        water_masks = cond['water_masks']
        water_caption = cond['water_caption']
        img_h, img_w = cond['size']
        bs = len(caption)

        obj_vectors = np.zeros((bs, self.max_obj), dtype=np.uint8)
        obj_masks = np.zeros((bs, self.max_obj, img_h, img_w))
        obj_boxes = np.zeros((bs, self.max_obj, 4))
        obj_attrs = classes

        for i in range(len(boxes)):
            for j in range(len(boxes[i])):
                box = boxes[i][j]
                pts = np.array([[box[0], box[1]], [box[0], box[3]], [box[2], box[3]], [box[2], box[1]]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                region = cv2.fillPoly(np.zeros((img_h, img_w)), [pts], 1)
                obj_boxes[i,j,:] = np.array(box)
                obj_masks[i,j,:,:] = region
                obj_vectors[i,j] = 1
            
            while len(obj_attrs[i]) < self.max_obj:
                obj_attrs[i].append('')
        
        obj_boxes[..., 0:2] /= img_w  # 第0、1列
        obj_boxes[..., 2:4] /= img_h  # 第2、3列
        
        # text condition
        context = self.text_encoder.encode( caption )
        if self.uc != None:
            self.uc = self.text_encoder.encode( [""] * bs )

        # water condition
        water_context = self.text_encoder.encode( water_caption )
        wat_boxes = torch.from_numpy(np.stack(water_boxes, axis=0)[:,np.newaxis,:]).float()
        wat_boxes[..., 0:2] /= img_w  # 第0、1列
        wat_boxes[..., 2:4] /= img_h  # 第2、3列
        wat_masks = torch.from_numpy(np.stack(water_masks, axis=0)[:,np.newaxis,:,:]).float()
        wat_vectors = torch.ones((bs, 1))

        batch = {
        "obj_masks" : obj_masks,
        "obj_boxes" : obj_boxes,
        "obj_attrs"  : obj_attrs,
        "obj_vectors" : obj_vectors,
        "wat_masks" : wat_masks,
        "wat_boxes" : wat_boxes,
        "wat_vectors" : wat_vectors
        }

        batch = batch_to_device(batch, 'cuda') 

        input = dict(
                x = None, 
                # x = torch.randn(bs, 4, img_w//8, img_h//8).to('cuda'), 
                timesteps = None, 
                context = context, 
                grounding_input = self.grounding_tokenizer_input.prepare(batch, water_context),
                inpainting_extra_input = None,
                grounding_extra_input = None,
            )

        samples_fake = self.sampler.sample(
            S=self.steps, 
            shape=(
                bs, 
                self.model.in_channels, 
                self.model.image_size, 
                self.model.image_size), 
            input=input,  
            uc=self.uc, 
            guidance_scale=self.guidance_scale)
        samples_fake = self.autoencoder.decode(samples_fake)
        samples = torch.clamp(samples_fake, min=-1, max=1) * 0.5 + 0.5

        for i in range(samples.shape[0]):
            sample = samples[i].cpu().numpy().transpose(1, 2, 0) * 255 
            sample = Image.fromarray(sample.astype(np.uint8))
            sample.save(os.path.join(self.image_dir, f'{cond["file_name"][i][:-4]}.jpg'))

            sample_w_box = draw_bounding_boxes(sample, boxes[i], classes[i], False)
        
            sample_w_box.save(os.path.join(self.image_w_box_dir, f'{cond["file_name"][i][:-4]}.jpg'))
            
def main(args, cond):
    # rand seed
    if args.rand_seed:
        torch.manual_seed(args.rand_seed)
    
    # load model
    process = Processer(args)


    process.generator(cond)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # data_args
    parser.add_argument("--data_path", type=str,  default="../ShipGD_val", help="data path")
    parser.add_argument("--out_path", type=str,  default="./out", help="output path")
    parser.add_argument("--mode", type=str,  default="val", help="")
    

    parser.add_argument("--rand_seed", type=int,  default=123, help="")
    parser.add_argument("--batch_size", type=int,  default=4, help="")
    parser.add_argument("--num_workers", type=int,  default=8, help="")
    parser.add_argument("--steps", type=int,  default=50, help="")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='', help="")

    parser.add_argument("--sd_ckpt", type=str,  default="./ckpt/X2Mari.pth", help="rand seed")
    
    args = parser.parse_args()

    cond = dict(
        caption=['An image of a river scene taken from a shore angle during the daytime. The weather appears to be clear with ample sunlight, suggesting good visibility.']*args.batch_size,
        boxes = [[[308, 135, 512, 226]]]*args.batch_size,
        classes=[['ship']]*args.batch_size,
        water_boxes=[[0, 133, 512, 430]]*args.batch_size,
        water_masks=["3014, 58, 425, 24, 5, 58, 361, 29, 4, 26, 4, 25, 2, 1, 2, 58, 331, 29, 65, 24, 5, 58, 272, 24, 1, 1, 3, 25, 4, 60, 5, 25, 4, 25, 5, 113, 152, 360, 5, 24, 91, 1, 1, 25, 4, 360, 1, 1, 4, 24, 5, 20, 4, 31, 4, 24, 2, 1, 2, 25, 6, 358, 1, 1, 4, 24, 26, 1, 34, 2, 31, 24, 6, 358, 6, 24, 5, 20, 5, 29, 5, 24, 6, 24, 6, 93503, 4, 508, 5, 507, 5, 507, 5, 507, 5, 492, 7, 1, 3, 1, 8, 2, 3, 11, 2, 474, 11, 1, 13, 10, 4, 473, 21, 1, 3, 9, 5, 473, 24, 10, 6, 472, 24, 10, 2, 1, 3, 472, 39, 1, 2, 1, 1, 1, 5, 2, 3, 457, 47, 1, 7, 457, 44, 1, 10, 457, 54, 458, 55, 457, 52, 1, 5, 37, 3, 2, 5, 1, 1, 1, 1, 1, 8, 394, 59, 35, 3, 1, 21, 393, 60, 34, 25, 393, 59, 35, 25, 393, 59, 35, 25, 393, 64, 5, 2, 2, 1, 2, 3, 5, 1, 1, 3, 5, 23, 1, 3, 1, 1, 389, 65, 4, 15, 1, 4, 5, 30, 388, 63, 6, 20, 5, 30, 388, 65, 5, 4, 1, 1, 1, 47, 388, 65, 4, 3, 1, 50, 389, 67, 1, 55, 1, 1, 1, 1, 385, 69, 1, 59, 383, 126, 1, 1, 384, 127, 385, 129, 383, 64, 1, 63, 384, 129, 383, 129, 383, 129, 383, 129, 383, 128, 384, 128, 384, 129, 383, 128, 384, 128, 384, 129, 383, 129, 383, 129, 383, 129, 383, 129, 383, 130, 1, 3, 378, 134, 378, 133, 379, 134, 378, 133, 379, 139, 373, 139, 373, 139, 373, 139, 373, 139, 373, 136, 376, 136, 376, 136, 376, 136, 376, 136, 376, 136, 376, 136, 376, 136, 376, 136, 376, 137, 11, 5, 163, 3, 1, 1, 1, 2, 1, 1, 186, 137, 11, 5, 163, 10, 186, 138, 11, 4, 163, 10, 186, 137, 11, 5, 163, 10, 186, 138, 11, 4, 163, 10, 186, 138, 1, 5, 1, 6, 1, 1, 158, 2, 1, 12, 1, 7, 1, 1, 176, 152, 159, 25, 176, 153, 158, 25, 176, 153, 158, 25, 176, 153, 158, 24, 177, 154, 2, 2, 153, 24, 2, 3, 1, 3, 32, 1, 1, 2, 2, 1, 2, 2, 1, 1, 55, 2, 2, 1, 2, 1, 4, 3, 53, 154, 1, 3, 153, 24, 1, 4, 1, 4, 30, 15, 54, 3, 1, 6, 2, 3, 53, 158, 153, 28, 1, 6, 29, 15, 54, 30, 38, 157, 154, 35, 29, 15, 54, 15, 53, 158, 153, 35, 29, 15, 54, 30, 38, 159, 1, 1, 2, 1, 2, 2, 140, 1, 1, 44, 2, 1, 1, 3, 5, 5, 1, 23, 13, 1, 3, 3, 1, 3, 1, 7, 3, 1, 1, 3, 4, 2, 1, 3, 1, 1, 1, 228, 1, 3, 138, 55, 5, 5, 1, 23, 10, 22, 3, 5, 4, 241, 138, 45, 1, 9, 4, 29, 5, 7, 1, 4, 1, 23, 4, 2, 1, 238, 138, 54, 5, 42, 1, 26, 5, 241, 138, 54, 6, 68, 6, 240, 134, 59, 1, 318, 133, 379, 133, 379, 133, 379, 133, 383, 126, 1, 2, 384, 125, 387, 123, 388, 124, 388, 124, 393, 17, 1, 1, 1, 43, 2, 45, 1, 2, 1, 1, 3, 1, 391, 1, 18, 1, 47, 44, 468, 44, 424, 39, 5, 44, 1250, 0"]*args.batch_size,
        water_caption=["The water surface is calm with a light blue-green hue."]*args.batch_size,
        file_name=[f'img_{str(i)}.jpg' for i in range(args.batch_size)],
        size=(512, 512)
    )
    if isinstance(cond['water_masks'][0], str):
        masks = []
        for box, mask in zip(cond['water_boxes'], cond['water_masks']):
            cvat_rle=dict(rle=mask,
                          top=box[1],
                          left=box[0],
                          width=box[2]-box[0])
            img_w, img_h = cond['size']
            masks.append(BaseData._rle2mask(None, cvat_rle, img_h, img_w))
        cond['water_masks'] = masks

    main(args, cond)

    