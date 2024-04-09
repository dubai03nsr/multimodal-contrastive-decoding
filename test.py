# /home1/09896/victorwang37/.cache/huggingface/modules/transformers_modules/openbmb/MiniCPM-V/bec7d1cd1c9e804c064ec291163e40624825eaaa/

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

print('cuda available', torch.cuda.is_available())
model = AutoModel.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(device='cuda:0', dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True)
model.eval()

image = Image.open('sample_imgs_3/0004886b7d043cfd_shrunk.jpg').convert('RGB')
src_text = '摆在一张棕色木桌子上的一个金色的首饰盒，旁边还有其他的金物件'

model.Chat(
    image=image,
    src_text=src_text,
    tokenizer=tokenizer,
    tgt_lang='en'
)

