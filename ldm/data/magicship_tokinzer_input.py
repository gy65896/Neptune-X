import torch as th 
from transformers import CLIPProcessor, CLIPModel

class GroundingNetInput:
    def __init__(self, version = "openai/clip-vit-large-patch14"):
        self.set = False 
        
        self.model = CLIPModel.from_pretrained(version).cuda()
        self.processor = CLIPProcessor.from_pretrained(version)
    
    def get_embedding(self, input):
        inputs = self.processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = th.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = self.model(**inputs)
        feature = outputs.text_model_output.pooler_output
        return feature

    def prepare(self, batch, wat_attrs):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the 
        input only for the ground tokenizer. 
        """

        self.set = True

        wat_cond = dict(wat_masks=batch['wat_masks'],
                             wat_boxes=batch['wat_boxes'],
                             wat_attrs=wat_attrs,
                             wat_vectors=batch['wat_vectors'])
        
        obj_text=[]
        for text in batch['obj_attrs']:
            obj_text.append(self.get_embedding(text).detach())
        obj_attrs=th.stack(obj_text)
        
        obj_cond = dict(obj_masks=batch['obj_masks'],
                             obj_boxes=batch['obj_boxes'],
                             obj_attrs=obj_attrs,
                             obj_vectors=batch['obj_vectors'])

        self.batch, self.max_box, self.in_dim = obj_attrs.shape
        _, self.wat_dim = batch['wat_vectors'].shape
        _, self.wat_chn, _ = wat_attrs.shape
        _, _, self.w, self.h = batch['obj_masks'].shape
        self.device = obj_attrs.device
        self.dtype = obj_attrs.dtype

        return {"wat_cond": wat_cond, 
                "obj_cond": obj_cond}


    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference, 
        please define the null input for the grounding tokenizer 
        """

        assert self.set, "not set yet, cannot call this funcion"
        batch =  self.batch  if batch  is None else batch
        device = self.device if device is None else device
        dtype = self.dtype   if dtype  is None else dtype

        wat_cond = dict(
            wat_masks=th.zeros(batch, 1, self.w, self.h).type(dtype).to(device),
            wat_boxes=th.zeros(batch, 1, 4).type(dtype).to(device),
            wat_attrs=th.zeros(batch, self.wat_chn, self.in_dim).type(dtype).to(device),
            wat_vectors=th.zeros(batch, self.wat_dim).type(dtype).to(device))
        
        obj_cond = dict(
            obj_masks=th.zeros(batch, self.max_box, self.w, self.h).type(dtype).to(device),
            obj_boxes=th.zeros(batch, self.max_box, 4).type(dtype).to(device),
            obj_attrs=th.zeros(batch, self.max_box, self.in_dim).type(dtype).to(device),
            obj_vectors=th.zeros(batch, self.max_box).type(dtype).to(device))

        return {"wat_cond": wat_cond, 
                "obj_cond": obj_cond}

if __name__=='__main__':
    model = GroundingNetInput()
    out = model.get_embedding(['a'])
    print(out.shape)








