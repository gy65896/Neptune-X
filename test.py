
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse, torch, os
from torch.utils.data import DataLoader

from trainer import batch_to_device, wrap_loader
from ldm.data.base import collate_fn, draw_bounding_boxes
from ldm.data.dataset_test import ShipGDData
from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler

device = "cuda" if torch.cuda.is_available() else "cpu"

class Processer:
    def __init__(self, args,):

        self.batch_size = args.batch_size
        self.guidance_scale = args.guidance_scale
        self.negative_prompt = args.negative_prompt
        
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
    def generator(self, batch, batch_here):

        boxes, classes = [], []
        for i in range(batch['obj_vectors'].shape[0]):
            box, class_ = [], []
            for j in range(batch['obj_vectors'].shape[1]):
                if batch['obj_vectors'][i, j] != 0:
                    class_.append(batch['obj_classes'][i][j])
                    label_box = [
                        batch['obj_boxes'][i][j][0],
                        batch['obj_boxes'][i][j][1],
                        batch['obj_boxes'][i][j][2],
                        batch['obj_boxes'][i][j][3]]
                    label_box = list(map(float, label_box))
                    box.append(label_box)
            classes.append(class_)
            boxes.append(box)
            
        
        # encoding_condition
        context = self.text_encoder.encode(  batch["caption"]  )
        wat_attrs = self.text_encoder.encode(  batch["wat_attrs"]  )
        grounding_input = self.grounding_tokenizer_input.prepare(batch, wat_attrs)

        input = dict( 
            x=None, 
            timesteps=None, 
            context=context, 
            inpainting_extra_input=None,
            grounding_extra_input=None,
            grounding_input=grounding_input 
        )

        # generation process
        samples_fake = self.sampler.sample(
            S=self.steps, 
            shape = (batch_here, 
                    self.model.in_channels, 
                    self.model.image_size, 
                    self.model.image_size),
            input=input,  
            uc=self.uc, 
            guidance_scale=self.guidance_scale
        )

        samples_fake = self.autoencoder.decode(samples_fake)

        samples = torch.clamp(samples_fake, min=-1, max=1) * 0.5 + 0.5

        # save img
        for i in range(samples.shape[0]):
            sample = samples[i].cpu().numpy().transpose(1, 2, 0) * 255 
            sample = Image.fromarray(sample.astype(np.uint8))
            sample.save(os.path.join(self.image_dir, f'{batch["file_name"][i][:-4]}.jpg'))

            sample_w_box = draw_bounding_boxes(sample, boxes[i], classes[i])
        
            sample_w_box.save(os.path.join(self.image_w_box_dir, f'{batch["file_name"][i][:-4]}.jpg'))
            
def main(args):
    # rand seed
    if args.rand_seed:
        torch.manual_seed(args.rand_seed)

    # load data
    dataset = ShipGDData(
        args.data_path, 
        args.mode)
    dataloader = DataLoader( 
        dataset,  
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True,
        collate_fn=collate_fn)
    
    # load model
    process = Processer(args)
    
    dataloader = wrap_loader(dataloader)

    for idx in tqdm(range(1+len(dataset)//args.batch_size)):

        batch = next(dataloader)
        batch_to_device(batch, device)

        process.generator(batch, args.batch_size)


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

    main(args)

    