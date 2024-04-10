# /home1/09896/victorwang37/.cache/huggingface/modules/transformers_modules/openbmb/MiniCPM-V/bec7d1cd1c9e804c064ec291163e40624825eaaa/
# 9d9376bea053209273767588969282a9f3ef95c0/

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import time

print('cuda available', torch.cuda.is_available())
model = AutoModel.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(device='cuda:0', dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True)
model.eval()

start = time.time()

# image = Image.open('shrunk-5/0004886b7d043cfd.jpg').convert('RGB')
image = Image.open('shrunk-5/000411001ff7dd4f.jpg').convert('RGB')
# src_text = '摆在一张棕色木桌子上的一个金色的首饰盒，旁边还有其他的金物件'
src_text = '在山里中站着两只鸡，一只黄色另一只黑黄色，它们俩站着看向同一个方向'

for i in range(11):
    hp = i * 0.1
    model.Chat(
        image=image,
        src_text=src_text,
        tokenizer=tokenizer,
        tgt_lang='en',
        txt_hp=hp,
        img_hp=hp
    )

print(time.time() - start)
