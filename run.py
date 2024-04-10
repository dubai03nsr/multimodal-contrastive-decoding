import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import os
import pickle

# load model
print('cuda available', torch.cuda.is_available())
model = AutoModel.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(device='cuda:0', dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True)
model.eval()

# read captions
cap_jsonl = pd.read_json('captions.jsonl', lines=True)
def get_caption(img_id, lang, idx=0):
    return cap_jsonl[cap_jsonl['image/key'] == img_id][lang].iloc[0]['caption'][idx]

dire = 'shrunk-5/'
dirs = [('en', 'zh'), ('zh', 'en')]
n_dir = len(dirs)
# direction -> {hp_i -> {img_id -> res}}
res = {d: {hp_i: {} for hp_i in range(11)} for d in dirs}

# run
for fname in os.listdir(dire):
    image = Image.open(dire + fname).convert('RGB')
    img_id = fname[:-4]

    for dirn in dirs:
        src_lang, tgt_lang = dirn
        src_text = get_caption(img_id, src_lang)

        for hp_i in range(11):
            hp = hp_i * 0.1
            res_run = model.Chat(
                image=image,
                src_text=src_text,
                tokenizer=tokenizer,
                tgt_lang=tgt_lang,
                txt_hp=hp,
                img_hp=hp
            )
            res[dirn][hp_i][img_id] = res_run

with open('save/run.pkl', 'wb') as f:
    pickle.dump(res, f)
