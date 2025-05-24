import os, cv2
from PIL import Image
import numpy as np
import random
from .base import ResizeData, DrawData

class ShipGDData(ResizeData, DrawData):
    def __init__(self, 
        base_path, 
        image_path: str = 'JPEGImages',
        mask_path: str = 'Annotations/mask',
        json_path: str = 'Annotations/json',
        data_path: str = 'ImageSets',
        mode: str = 'train', 
        resolution: tuple = (512, 512),
        max_length: int = 30,
        comp_ratio: float = 1.0, 
        dropout_prob: float = 0.1, 
        category_dict: dict = {"ship": 0, "buoy": 1, "person": 2, 
                               "floating object": 3, "fixed object": 4, '': -1}, 
        rand_crop: bool = True,
        rand_flip: bool = True,
        wat_mask: str='refine', # rand, coarse (box), refine (mask)
        display: bool = False,
        save_dir: str = './out_img', 
        water_colors: np.ndarray=None,
        obj_colors: np.ndarray=None,
        alpha: float = 0.3, 
        thickness: int = 1,
        font_scale: float = 1, 
        left_t: int = 5, 
        top_t: int = 25, 
        line_h: int = 30):

        self.mode = mode
        with open(os.path.join(base_path, f'{data_path}/{mode}.txt'), 'r') as file:
            self.files = file.read().splitlines()
        self.json_path = os.path.join(base_path, json_path)
        self.image_path = os.path.join(base_path, image_path)
        self.mask_path = os.path.join(base_path, mask_path)
        
        self.resolution = resolution
        self.comp_ratio = comp_ratio
        self.rand_crop = rand_crop
        self.rand_flip = rand_flip
        self.max_length = max_length
        self.dropout_prob = dropout_prob
        self.wat_mask = wat_mask
        
        self.category_dict = category_dict

        # debug
        self.display = display
        self.water_colors = water_colors
        self.obj_colors = obj_colors
        self.alpha = alpha
        self.thickness = thickness
        self.font_scale = font_scale
        self.left_t = left_t
        self.top_t = top_t
        self.line_h = line_h
        if display:
            os.makedirs(save_dir, exist_ok = True)
            self.save_dir = save_dir
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):

        data_name = self.files[idx]
        json_path = os.path.join(self.json_path, data_name + '.json')
        img_path = os.path.join(self.image_path, data_name + '.jpg')
        mask_path = os.path.join(self.mask_path, data_name + '.jpg')

        # load data
        anns = self.load_data(json_path)
        image = self._load_img(img_path, True)
        water_masks = self._load_img(mask_path, True) if os.path.exists(mask_path) \
            else self._rle2mask(anns['mask'], int(anns['height']), int(anns['width']))

        # adapt data
        anns, image, water_masks = self.adapt_data(anns, image, water_masks)

        # std data
        data = self.std_data(anns, image, water_masks)

        # display data
        self._disply(**data) if self.display else None

        return self._dict2tensor(data)
    
    def std_data(self, anns, image, wat_masks_ori):
        # init
        height, width = int(anns['height']), int(anns['width'])
        wat_vectors = np.zeros(1)
        obj_masks = np.zeros((self.max_length, height, width))
        obj_boxes = np.zeros((self.max_length, 4))
        obj_vectors = np.zeros(self.max_length)
        obj_classes, obj_attrs = [], []
        wat_masks = np.zeros((1, height, width))
        wat_boxes = np.zeros((1, 4))
        
        # load txt condition
        image = (image * 2.0 - 1.0).transpose(2, 0, 1)
        caption = self._dropout_tool(anns['caption'])

        # load water condition
        wat_box_norm, wat_box = self._mask2box(wat_masks_ori)
        wat_boxes[0] = np.array(wat_box_norm)
        if self.wat_mask == 'rand':
            if random.random() < 0.5:
                wat_masks[0] = self._box2mask(wat_box, wat_masks_ori.shape)
                wat_vectors[0] = 0.5
            else:
                wat_masks[0] = wat_masks_ori
                wat_vectors[0] = 1
        elif self.wat_mask == 'coarse':
            wat_masks[0] = self._box2mask(wat_box, wat_masks_ori.shape)
            wat_vectors[0] = 0.5
        elif self.wat_mask == 'refine':
            wat_masks[0] = wat_masks_ori
            wat_vectors[0] = 1

        wat_attrs = self._dropout_tool(anns['mask']['attribute'])

        # load box condition
        i = 0
        for i in range(len(anns['box'])):
            if i > (self.max_length - 1):
                break
            box, mask = self._load_box(anns['box'][i], height, width, True, True)
            obj_masks[i,:,:] = mask
            obj_boxes[i,:] = box
            obj_classes.append(anns['box'][i]['category'])
            obj_attrs.append(f"<{anns['box'][i]['category']}> {self._dropout_tool(anns['box'][i]['attribute'])}")
            obj_vectors[i] = 1
        
        while len(obj_classes) < self.max_length:
            obj_classes.append('')
            obj_attrs.append('')

        obj_idxes = np.stack(map(lambda x: self.category_dict.get(x, -1), obj_classes))

        return dict(file_name = anns['name'], image = image, caption = caption,
                    wat_masks = wat_masks, wat_boxes = wat_boxes, wat_attrs = wat_attrs, wat_vectors = wat_vectors,
                    obj_masks = obj_masks, obj_boxes = obj_boxes, obj_classes = obj_classes,
                    obj_attrs = obj_attrs, obj_idxes = obj_idxes, obj_vectors = obj_vectors)
    
    def load_data(self,
                  json_path: str=None,
                  idx: int=0) -> dict:
        if json_path==None:
            json_path = os.path.join(self.json_path, self.files[idx] + '.json')
            print(f'load data from {json_path}.')
        return self._load_json(json_path)
    
    def save_mask(self,
                  idx: int=0) -> dict:
        os.makedirs(self.mask_path, exist_ok = True)

        data_name = self.files[idx]
        json_path = os.path.join(self.json_path, data_name + '.json')
        save_file = os.path.join(self.mask_path, data_name + '.jpg')

        anns = self.load_data(json_path)

        mask = (self._rle2mask(anns['mask'], int(anns['height']), 
                               int(anns['width']))*255).astype(np.uint8)
        cv2.imwrite(save_file, mask)
        print(f'save mask to {save_file}.')
    
    def save_obj(self, 
                  obj_path: str, 
                  idx: int=0):

        os.makedirs(obj_path, exist_ok = True)
        for i in self.category_dict.keys():
            os.makedirs(obj_path+'/'+i, exist_ok = True)
        
        data_name = self.files[idx]
        json_path = os.path.join(self.json_path, data_name + '.json')
        img_path = os.path.join(self.image_path, data_name + '.jpg')
        
        boxes = self.load_data(json_path)['box']
        image = self._load_img(img_path, True)

        for i in range(len(boxes)):
            save_file = f"{obj_path}/{boxes[i]['category']}/{data_name}_{i}.jpg"
            xtl, ytl, xbr, ybr =self._load_box(boxes[i])

            obj = Image.fromarray((image[ytl: ybr, xtl: xbr, :] * 255.).astype(np.uint8))
            obj.save(save_file)
            print(f'save obj to {save_file}.')

if __name__ == "__main__":
    
    bash_paths = '../ShipGD/'
    resolution = (512, 768)

    dataset = ShipGDData(bash_paths, resolution=resolution, mode='train', display=True)

    for idx in range(len(dataset)):
        data = dataset[idx]
        print(data['file_name'])

        