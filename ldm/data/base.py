import os, json, cv2, torch
from PIL import Image, ImageDraw
import numpy as np
import random
from torch.utils.data import Dataset
from collections import defaultdict
from copy import deepcopy

def draw_bounding_boxes(
        img, 
        boxes,
        names, 
        is_normalized=True, 
        colors=["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"], 
        thickness=1):

    img_new = deepcopy(img)
    draw = ImageDraw.Draw(img_new)
    
    W, H = img.size
    
    for bid, box in enumerate(boxes):
        xtl, ytl, xbr, ybr = box[0], box[1], box[2], box[3]
        
        if is_normalized:
            xtl = int(xtl * W)
            ytl = int(ytl * H)
            xbr = int(xbr * W)
            ybr = int(ybr * H)
        else:
            xtl, ytl, xbr, ybr = map(int, [xtl, ytl, xbr, ybr])
        
        xtl = max(0, min(xtl, W-1))
        ytl = max(0, min(ytl, H-1))
        xbr = max(0, min(xbr, W-1))
        ybr = max(0, min(ybr, H-1))

        draw.rectangle([xtl, ytl, xbr, ybr], outline=colors[bid % len(colors)], width=thickness)
        draw.text((xtl, ytl), names[bid], fill=colors[bid % len(colors)])
    return img_new

def to_numpy(val: torch.Tensor):
    return (val * 0.5 + 0.5).detach().cpu().numpy().transpose(1, 2, 0)

def to_tensor(val: np.ndarray):
    return torch.Tensor((val * 2.0 - 1.0).transpose(2, 0, 1))

def exist(val):
    return val is not None

def collate_fn(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - obj: LongTensor of shape (N, L) giving object categories
    - masks: FloatTensor of shape (N, L, H, W)
    - is_valid_obj: FloatTensor of shape (N, L)
    """
    all_meta_data = defaultdict(list)

    for i, meta_data in enumerate(batch):
        for key, value in meta_data.items():
            all_meta_data[key].append(value)

    for key, value in all_meta_data.items():
        if isinstance(value[0], torch.Tensor) or isinstance(value[0], torch.LongTensor):
            all_meta_data[key] = torch.stack(value)

    return all_meta_data

class BaseData(Dataset):
    def __init__(self, 
        mode: str = 'train', 
        dropout_prob: int = 0.1, 
        resolution: tuple = (512, 512)):
        
        self.mode = mode
        
        self.dropout_prob = dropout_prob
        self.resolution = resolution

    def _load_json(self, json_name: str) -> dict: 
        with open(json_name, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_img(self, img_name: str, norm_img: bool=False) -> np.ndarray:
        img = np.array(Image.open(img_name))
        if norm_img:
            img = img / 255
        return img
    
    def _load_npy(self, npy_name: str) -> np.ndarray:
        img = np.load(npy_name)
        return img
    
    def _load_box(self, 
                  obj: dict, 
                  img_h: int=None, 
                  img_w: int=None, 
                  norm_box: bool=False, 
                  get_region: bool=False) -> np.ndarray:
        box = [obj['xtl'], obj['ytl'], obj['xbr'], obj['ybr']]
        box = list(map(lambda x: int(float(x)) if not isinstance(x, int) else x, box))
        if get_region:
            pts = np.array([[box[0], box[1]], [box[0], box[3]], [box[2], box[3]], [box[2], box[1]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            region = cv2.fillPoly(np.zeros((img_h, img_w), dtype=np.float32), [pts], 1)
        if norm_box:
            box = [box[0]/img_w, box[1]/img_h, box[2]/img_w, box[3]/img_h]
        try:
            return box, region
        except:
            return box

    def _dropout_tool(self, prompt: str=None) -> str:
        if 'train' in self.mode:
            if random.random() < self.dropout_prob:
                return 0 if prompt==None else ''
        elif 'gen' in self.mode:
            return 0 if prompt==None else ''
        return 1 if prompt==None else prompt
    
    def _xyxy2xywh(self, box: list) -> list:
        x0, y0, x1, y1 = box
        return [x0, y0, x1-x0, y1-y0]
    
    def _xywh2xyxy(self, box: list) -> list:
        x0, y0, w, h = box
        return [x0, y0, x0+w, y0+h]
    
    def _xyxy2centerwh(self, box: list) -> list:
        x0, y0, x1, y1 = box
        return [(x0+x1)//2, (y0+y1)//2, x1-x0, y1-y0]
    
    def _centerwh2xyxy(self, box: list) -> list:
        x, y, w, h = box
        return [x-w//2, y-h//2, x+w//2, y+h//2]
    
    def _mask2box(self, mask: np.ndarray, norm_box=True) -> list:
        h, w = mask.shape
        y_nonzero, x_nonzero = np.nonzero(mask)
        if len(x_nonzero) > 0 and len(y_nonzero) > 0:
            x0 = x_nonzero.min()
            x1 = x_nonzero.max()
            y0 = y_nonzero.min()
            y1 = y_nonzero.max()
            if norm_box:
                return [x0/w, y0/h, x1/w, y1/h], [x0, y0, x1, y1]
            else:
                return [x0, y0, x1, y1]
        else:
            if norm_box:
                return [0, 0, 0, 0], [0, 0, 0, 0]
            else:
                return [0, 0, 0, 0]
        
    def _box2mask(self, box: list, mask_shape: tuple) -> np.ndarray:
        return cv2.rectangle(np.zeros(mask_shape), (box[0], box[1]), 
                      (box[2], box[3]), 1, thickness=cv2.FILLED)
    
    def _mask2xywh(self, mask: np.ndarray):
        
        rows, cols = np.where(mask)
        if len(rows) == 0:
            raise ValueError("False Mask")
        
        top = np.min(rows)
        bottom = np.max(rows)
        left = np.min(cols)
        right = np.max(cols)
        height = bottom - top + 1
        width = right - left + 1
        return top, bottom, left, right, height, width
        
    def _rle2mask(self, cvat_rle: dict, img_h: int, img_w: int) -> np.ndarray:
        rle = list(map(int, cvat_rle.get('rle').split(',')))
        left = int(cvat_rle.get('left'))
        top = int(cvat_rle.get('top'))
        width = int(cvat_rle.get('width'))
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        value = 0
        offset = 0
        for rle_count in rle:
            while rle_count > 0:
                y, x = divmod(offset, width)
                mask[y + top][x + left] = value
                rle_count -= 1
                offset += 1
            value = 1 - value
        return mask
    
    def _mask2rle(self, mask: np.ndarray) -> dict:
        top, bottom, left, right, height, width = self._mask2xywh(mask)
    
        region = mask[top:bottom+1, left:right+1].astype(np.uint8)
    
        flat = region.ravel()
        rle = []
        current_val = 0
        count = 0
    
        if flat.size > 0 and flat[0] == 1:
            rle.append(0)
            current_val = 1
    
        for pixel in flat:
            if pixel == current_val:
                count += 1
            else:
                rle.append(count)
                current_val = 1 - current_val
                count = 1
        if count > 0:
            rle.append(count)
    
        if not rle:
            rle = [0]
        elif len(rle) % 2 != 0:
            rle.append(0)
    
        return {
        'rle': ', '.join(map(str, rle)),
        'left': str(left),
        'top': str(top),
        'width': str(width),
        'height': str(height)}
    
    def _dict2tensor(self, data: dict):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = torch.Tensor(value)
        return data

class DrawData(BaseData):
    def __init__(self, 
        save_dir: str='.',
        water_colors: np.ndarray=None,
        obj_colors: np.ndarray=None,
        alpha: float = 0.3, 
        thickness: int = 1,
        font_scale: float = 1, 
        left_t: int = 5, 
        top_t: int = 25, 
        line_h: int = 30):
        super().__init__()

        self.save_dir = save_dir

        self.water_colors = water_colors
        self.obj_colors = obj_colors
        self.alpha = alpha
        self.thickness = thickness
        self.font_scale = font_scale
        self.left_t = left_t
        self.top_t = top_t
        self.line_h = line_h
        
    def _draw_mask(self, 
        image: np.ndarray, 
        masks: np.ndarray,
        wat_vectors: np.ndarray) -> (np.ndarray):
        self.water_colors = self.water_colors if \
            exist(self.water_colors)else np.random.randint(0, 255, (3))

        overlay = image.copy()
        overlay[masks[0] == 1] = self.water_colors
        alpha = 0 if wat_vectors[0] == 0 else self.alpha
        image = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
        return image
    
    def _draw_box(self, 
        image: np.ndarray, 
        boxes: np.ndarray, 
        classes: list, 
        vectors: np.ndarray, 
        shape: np.ndarray, 
        norm_box: bool=True) -> (np.ndarray):
        num = len(vectors)
        self.obj_colors = self.obj_colors if \
            exist(self.obj_colors) else np.random.randint(0, 255, (num, 3))

        for i in range(num):
            color = tuple(map(int, self.obj_colors[i]))
            if norm_box:
                start_point = (int(boxes[i][0] * shape[1]), int(boxes[i][1] * shape[0]))
                end_point = (int(boxes[i][2] * shape[1]), int(boxes[i][3] * shape[0]))
            else:
                start_point = (int(boxes[i][0]), int(boxes[i][1]))
                end_point = (int(boxes[i][2]), int(boxes[i][3]))
            cv2.rectangle(image, start_point, end_point, color, self.thickness)
            cv2.putText(image, classes[i], start_point, 
                cv2.FONT_HERSHEY_COMPLEX, self.font_scale, color, self.thickness)
        return image
    
    def _draw_txt(self, 
        caption: str, 
        wat_attrs: str,
        obj_attrs: list, 
        obj_vectors: np.ndarray, 
        shape: np.ndarray) -> (np.ndarray):
        image = np.ones(shape, dtype=np.uint8) * 255
        num = int(sum(obj_vectors)) + 2
        self.obj_colors = self.obj_colors if \
            exist(self.obj_colors) else np.random.randint(0, 255, (num, 3))
        colors_new = np.concatenate([np.zeros((1, 3)), 
                    [self.water_colors], self.obj_colors], 0)
        txt_new = [caption] + [f'<water surface> {wat_attrs}'] + obj_attrs

        top_t = self.top_t
        for i in range(num):
            color = tuple(map(int, colors_new[i]))
            words = txt_new[i].split()
            lines = []
            line = ""
            for word in words:
                if cv2.getTextSize(line + word, cv2.FONT_HERSHEY_COMPLEX,
                    self.font_scale, self.thickness)[0][0] < shape[1] - self.left_t:
                    line += word + " "
                else:
                    lines.append(line)
                    line = word + " "
            lines.append(line)
            for line in lines:
                image = cv2.putText(image, line, (self.left_t, top_t), 
                    cv2.FONT_HERSHEY_COMPLEX, self.font_scale, color, self.thickness)
                top_t += self.line_h
        return image
    
    def _disply(self, 
        file_name: str = None, 
        image: np.ndarray = None, 
        caption: str = None, 
        wat_masks: np.ndarray = None, 
        wat_attrs: str = None, 
        wat_vectors: np.ndarray = None,
        obj_boxes: np.ndarray = None,
        obj_classes: list = None, 
        obj_attrs: list = None, 
        obj_vectors: np.ndarray = None, 
        shape: np.ndarray = None,
        return_type = False,
        **kwargs):
        
        if not exist(image):
            image = np.ones(shape, dtype=np.uint8) * 255
        else:
            image = ((image * 0.5 + 0.5) * 255).transpose(1, 2, 0).astype(np.uint8)
            shape = image.shape

        if exist(wat_masks):
            image = self._draw_mask(image, wat_masks, wat_vectors)
            
        if exist(obj_boxes) and exist(obj_classes):
            image = self._draw_box(image, obj_boxes, obj_classes, obj_vectors, shape)

        if exist(caption) and exist(wat_attrs) and exist(obj_attrs):
            image_txt = self._draw_txt(caption, wat_attrs, obj_attrs, obj_vectors, shape)
            # masks_new = np.repeat((masks.transpose(1,2,0) * 255).astype(np.uint8), 3, axis=2)
            image = np.concatenate([image, image_txt], 1)

        if return_type == 'numpy':
            return image / 255.
        elif return_type == 'tensor':
            return torch.Tensor(((image / 255.) * 2.0 - 1.0).transpose(2, 0, 1))
        else:
            pil_image = Image.fromarray(image)
            pil_image.save(os.path.join(self.save_dir, file_name))
            print(f'save image to {file_name}.')

class ResizeData(BaseData):
    def __init__(self, 
        resolution: tuple = (512, 512),
        comp_ratio: float = 1.0,
        rand_crop: bool=True,
        rand_flip: bool=True):
        
        self.resolution = resolution
        self.comp_ratio = comp_ratio

        self.rand_crop = rand_crop
        self.rand_flip = rand_flip
    
    def _resize_img(self, 
                    img: np.ndarray, 
                    comp_size: tuple, 
                    x: int, 
                    y: int):
        img = cv2.resize(img, comp_size)
        if len(img.shape) == 3:
            img = img[y : y + self.resolution[0], x : x + self.resolution[1], :]
        elif len(img.shape) == 2:
            img = img[y : y + self.resolution[0], x : x + self.resolution[1]]
        else:
            raise ValueError("Error Image Channel!")
        return img
    
    def _resize_box(self, 
                    box: np.ndarray, 
                    comp_ratio: tuple, 
                    x: int, 
                    y: int):
        re_xtl = int(box[0] * comp_ratio[1])
        re_ytl = int(box[1] * comp_ratio[0])
        re_xbr = int(box[2] * comp_ratio[1])
        re_ybr = int(box[3] * comp_ratio[0])
        res_ori = (re_xbr - re_xtl) * (re_ybr - re_ytl)

        return [np.clip(re_xtl - x, 0, self.resolution[1]).astype(int),
                np.clip(re_ytl - y, 0, self.resolution[0]).astype(int),
                np.clip(re_xbr - x, 0, self.resolution[1]).astype(int),
                np.clip(re_ybr - y, 0, self.resolution[0]).astype(int)], max(res_ori, 1)
    
    def _rand_region(self, 
                     img_h: int, 
                     img_w: int):

        # 1. resize
        short_side = min((img_h, img_w))
        target_long = int(max(self.resolution) * self.comp_ratio)
    
        new_h = int(img_h * (target_long / img_w)) if img_w == short_side else target_long
        new_w = target_long if img_w == short_side else int(img_w * (target_long / img_h))
        
        comp_size = (new_w, new_h)
        comp_ratio = (new_w / img_w, new_h / img_h)
        
        # 2. crop
        max_x = new_w - self.resolution[1]
        max_y = new_h - self.resolution[0]

        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        return comp_ratio, comp_size, x, y

    def crop_data(self, 
                  anns: dict, 
                  image: np.ndarray, 
                  water_masks: np.ndarray):
        
        img_h, img_w = int(anns['height']), int(anns['width'])

        comp_ratio, comp_size, x, y = self._rand_region(img_h, img_w)

        image = self._resize_img(image, comp_size, x, y)
        water_masks = self._resize_img(water_masks, comp_size, x, y)
        
        anns['height'], anns['width'] = self.resolution[0], self.resolution[1]
        
        if 'box' in anns.keys():
            anns_box = []
            for obj in anns['box']:
                box = self._load_box(obj, img_h, img_w)
                box, res_ori = self._resize_box(box, comp_ratio, x, y)
                res_fin = int(box[2] - box[0]) * int(box[3] - box[1])
                if res_fin/res_ori>0.2 and res_fin>20:
                    obj['xtl'] = str(box[0])
                    obj['ytl'] = str(box[1])
                    obj['xbr'] = str(box[2])
                    obj['ybr'] = str(box[3])
                    anns_box.append(obj)
            anns['box'] = anns_box
        return anns, image, water_masks
    
    def flip_data(self, 
                  anns: dict, 
                  image: np.ndarray, 
                  water_masks: np.ndarray):
        if random.random() < 0.5:
            img_h, img_w = int(anns['height']), int(anns['width'])
            image = cv2.flip(image, 1)
            water_masks = cv2.flip(water_masks, 1)
            if 'box' in anns.keys():
                anns_box = []
                for obj in anns['box']:
                    box = self._load_box(obj, img_h, img_w)
                    x = (int(img_w - box[0]), int(img_w - box[2]))
                    obj['xtl'] = str(min(x))
                    obj['ytl'] = str(box[1])
                    obj['xbr'] = str(max(x))
                    obj['ybr'] = str(box[3])
                    anns_box.append(obj)
                anns['box'] = anns_box
        return anns, image, water_masks

    def adapt_data(self, 
                   anns: dict, 
                  image: np.ndarray, 
                  water_masks: np.ndarray):
        if self.rand_crop:
            anns, image, water_masks = self.crop_data(anns, image, water_masks)
        if self.rand_flip:
            anns, image, water_masks = self.flip_data(anns, image, water_masks)
        return anns, image, water_masks