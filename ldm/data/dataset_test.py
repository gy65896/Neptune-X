import os, json, torch
from PIL import Image
import numpy as np
from tqdm import tqdm 
from .base import BaseData


class ShipGDData(BaseData):
    def __init__(self, 
        base_path, 
        file_list,
        image_path: str = 'JPEGImages',
        mask_path: str = 'Annotations/mask',
        json_path: str = 'Annotations/json',
        data_path: str = 'ImageSets',
        max_length: int = 30,
        category_dict: dict = {"ship": 0, "buoy": 1, "person": 2, 
                               "floating object": 3, "fixed object": 4, '': -1}):

        with open(os.path.join(base_path, f'{data_path}/{file_list}.txt'), 'r') as file:
            self.files = file.read().splitlines()
        self.json_path = os.path.join(base_path, json_path)
        self.image_path = os.path.join(base_path, image_path)
        self.mask_path = os.path.join(base_path, mask_path)

        self.max_length = max_length
        self.category_dict = category_dict

    def __len__(self):
        return len(self.files)
    
    def _dict2tensor(self, data: dict):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = torch.Tensor(value)
        return data
    
    def __getitem__(self, idx):

        data_name = self.files[idx]
        json_path = os.path.join(self.json_path, data_name + '.json')
        mask_path = os.path.join(self.mask_path, data_name + '.jpg')

        # load data
        with open(json_path, 'r', encoding='utf-8') as f:
            anns = json.load(f)

        water_masks = np.array(Image.open(mask_path)) / 255

        # std data
        data = self.std_data(anns, water_masks)


        return self._dict2tensor(data)
    
    def std_data(self, anns, wat_masks_ori):
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
        caption = anns['caption']

        # load water condition
        wat_box_norm, _ = self._mask2box(wat_masks_ori)
        wat_boxes[0] = np.array(wat_box_norm)
        wat_masks[0] = wat_masks_ori
        wat_attrs = anns['mask']['attribute']
        wat_vectors[0] = 1

        # load box condition
        i = 0
        for i in range(len(anns['box'])):
            if i > (self.max_length - 1):
                break
            box, mask = self._load_box(anns['box'][i], height, width, True, True)
            obj_masks[i,:,:] = mask
            obj_boxes[i,:] = box
            obj_classes.append(anns['box'][i]['category'])
            obj_attrs.append(f"<{anns['box'][i]['category']}> {anns['box'][i]['attribute']}")
            obj_vectors[i] = 1
        
        while len(obj_classes) < self.max_length:
            obj_classes.append('')
            obj_attrs.append('')
        
        obj_idxes = np.stack(map(lambda x: self.category_dict.get(x, -1), obj_classes))

        return dict(file_name = anns['name'], caption = caption,
                    wat_masks = wat_masks, wat_boxes = wat_boxes, wat_attrs = wat_attrs, wat_vectors = wat_vectors,
                    obj_masks = obj_masks, obj_boxes = obj_boxes, obj_classes = obj_classes,
                    obj_attrs = obj_attrs, obj_idxes = obj_idxes, obj_vectors = obj_vectors)

if __name__ == "__main__":
    
    bash_paths = '../ShipGD_test/'

    dataset = ShipGDData(bash_paths, 'test.txt')

    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]

        