
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse, torch, os, cv2, json

from ldm.data.base import BaseData, draw_bounding_boxes
from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_cond_from_json(json_path, batch_size):
    """
    Load condition data from JSON file and expand to specified batch size
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Expand data to specified batch size
    cond = {
        'caption': data['caption'] * batch_size,
        'boxes': data['boxes'] * batch_size,
        'classes': data['classes'] * batch_size,
        'water_boxes': data['water_boxes'] * batch_size,
        'water_masks': data['water_masks'] * batch_size,
        'water_caption': data['water_caption'] * batch_size,
        'file_name': [f'img_{str(i)}.jpg' for i in range(batch_size)],
        'size': tuple(data['size'])
    }
    
    return cond

def draw_conditions_on_image(
    image,
    boxes,
    classes,
    water_box,
    water_mask,
    caption,
    water_caption,
    draw_text=True,
):
    """
    Draw all condition information on image: detection boxes, water mask, text conditions and water description
    """
    
    # Convert to PIL image
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image.copy()
    
    # Create drawing object
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load font, if failed use default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    
    
    # Draw water region
    if isinstance(water_mask, np.ndarray) and len(water_mask.shape) == 2:
        # Create colored mask with transparency
        colored_mask = np.zeros((water_mask.shape[0], water_mask.shape[1], 4), dtype=np.uint8)  # RGBA
        colored_mask[water_mask > 0] = [0, 255, 255, 128]  # Cyan with 50% transparency
        
        # Add mask to image with proper alpha blending
        mask_pil = Image.fromarray(colored_mask, 'RGBA')
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), mask_pil).convert('RGB')
        draw = ImageDraw.Draw(pil_image)

    # Draw detection boxes
    for i, (box, cls) in enumerate(zip(boxes, classes)):
        x1, y1, x2, y2 = box
        # Draw rectangle box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        # Draw class label
        draw.text((x1, y1-20), cls, fill='red', font=font)
    
    if draw_text:
        # Draw text information
        text_y = 10
        line_height = 20

        # Draw main description text
        caption_text = f"Caption: {caption[:50]}..."
        draw.text((10, text_y), caption_text, fill='white', font=small_font)
        text_y += line_height

        # Draw water description
        water_text = f"Water: {water_caption[:50]}..."
        draw.text((10, text_y), water_text, fill='cyan', font=small_font)
        text_y += line_height

        for i, (box, cls) in enumerate(zip(boxes, classes)):
            box_text = f"Box {i+1}: {cls} at {box}"
            draw.text((10, text_y), box_text, fill='red', font=small_font)
            text_y += line_height

        # Draw water box information
        water_box_text = f"Water Box: {water_box}"
        draw.text((10, text_y), water_box_text, fill='cyan', font=small_font)
        text_y += line_height
    
    return pil_image

def batch_to_device(batch, device, force_float32=True, use_fp16=False):

    if isinstance(batch, (np.ndarray, np.generic)):
        batch = torch.from_numpy(batch)
        if use_fp16 and batch.is_floating_point():
            batch = batch.to(torch.float16)
        elif force_float32 and batch.is_floating_point() and batch.dtype != torch.float32:
            batch = batch.to(torch.float32)
        batch = batch.to(device)
        return batch
    elif isinstance(batch, torch.Tensor):
        if use_fp16 and batch.is_floating_point():
            batch = batch.to(torch.float16)
        elif force_float32 and batch.is_floating_point() and batch.dtype != torch.float32:
            batch = batch.to(torch.float32)
        batch = batch.to(device)
        return batch
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device, force_float32, use_fp16) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        batch = [batch_to_device(v, device, force_float32, use_fp16) for v in batch]
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
        self.fp16 = args.fp16
        self.draw_text = True
        
        os.makedirs(args.out_path, exist_ok=True)
        self.image_dir = os.path.join(args.out_path,'image')
        os.makedirs(self.image_dir, exist_ok=True)
        self.image_w_cond_dir = os.path.join(args.out_path,'img_w_cond')
        os.makedirs(self.image_w_cond_dir, exist_ok=True)

        self.load_sd_ckpt(args.sd_ckpt, args.steps)

    def load_sd_ckpt(self, ckpt_path, steps):
    
        saved_ckpt = torch.load(ckpt_path, map_location=device)
        config = saved_ckpt["config_dict"]["_content"]

        model = instantiate_from_config(config['model']).to(device).eval()
        autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
        text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
        diffusion = instantiate_from_config(config['diffusion']).to(device)
        
        # Enable FP16 acceleration
        if self.fp16:
            model = model.half()
            autoencoder = autoencoder.half()
            text_encoder = text_encoder.half()
            diffusion = diffusion.half()
            torch.cuda.empty_cache()

        model.load_state_dict( saved_ckpt['model'] )
        autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
        text_encoder.load_state_dict( saved_ckpt["text_encoder"]  ,strict=False)
        diffusion.load_state_dict( saved_ckpt["diffusion"]  )

        grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
        model.grounding_tokenizer_input = grounding_tokenizer_input

        self.sampler = PLMSSampler(diffusion, model)
        self.steps = steps 
        self.text_encoder = text_encoder
        self.uc = self.text_encoder.encode( self.batch_size*[self.negative_prompt] )
        self.grounding_tokenizer_input = grounding_tokenizer_input
        self.model = model
        self.autoencoder = autoencoder
    
    @torch.no_grad()
    def generator(self, cond):
        # Use autocast for FP16 acceleration
        if self.fp16:
            with torch.cuda.amp.autocast():
                return self._generator_impl(cond)
        else:
            return self._generator_impl(cond)
    
    def _generator_impl(self, cond):
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
        
        obj_boxes[..., 0:2] /= img_w 
        obj_boxes[..., 2:4] /= img_h 
        
        # text condition
        context = self.text_encoder.encode( caption )
        if self.uc != None:
            self.uc = self.text_encoder.encode( [""] * bs )

        # water condition
        water_context = self.text_encoder.encode( water_caption )
        wat_boxes = torch.from_numpy(np.stack(water_boxes, axis=0)[:,np.newaxis,:]).float()
        wat_boxes[..., 0:2] /= img_w 
        wat_boxes[..., 2:4] /= img_h  
        
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

        # Save original water_masks for drawing
        original_water_masks = water_masks.copy()
        
        batch = batch_to_device(batch, 'cuda', use_fp16=self.fp16) 

        input = dict(
                x = None,
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

            # Draw conditions on image
            sample_w_cond = draw_conditions_on_image(
                sample, 
                boxes[i], 
                classes[i], 
                water_boxes[i], 
                original_water_masks[i], 
                caption[i], 
                water_caption[i],
                draw_text=self.draw_text,
            )
        
            sample_w_cond.save(os.path.join(self.image_w_cond_dir, f'{cond["file_name"][i][:-4]}.jpg'))
        
        # Clear GPU memory
        if self.fp16:
            torch.cuda.empty_cache()
            
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
    parser.add_argument("--json_path", type=str, default="./sample/1.json", help="path to JSON file containing condition data")
    parser.add_argument("--out_path", type=str,  default="./out", help="output path")

    parser.add_argument("--rand_seed", type=int,  default=64, help="")
    parser.add_argument("--batch_size", type=int,  default=4, help="")
    parser.add_argument("--num_workers", type=int,  default=8, help="")
    parser.add_argument("--steps", type=int,  default=50, help="")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='', help="")

    parser.add_argument("--sd_ckpt", type=str,  default="./ckpt/X2Mari.pth", help="rand seed")
    parser.add_argument("--fp16", action="store_true", help="use fp16 for faster inference")
        
    args = parser.parse_args()

    # Load condition data from JSON file
    cond = load_cond_from_json(args.json_path, args.batch_size)
    if isinstance(cond['water_masks'][0], str):
        # Decode RLE mask and replicate to all batches
        first_box = cond['water_boxes'][0]
        first_mask = cond['water_masks'][0]
        
        cvat_rle = dict(rle=first_mask,
                       top=first_box[1],
                       left=first_box[0],
                       width=first_box[2]-first_box[0])
        img_w, img_h = cond['size']
        
        decoded_mask = BaseData._rle2mask(None, cvat_rle, img_h, img_w)
        cond['water_masks'] = [decoded_mask] * args.batch_size

    main(args, cond)
