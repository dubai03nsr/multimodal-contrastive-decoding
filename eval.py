import pickle
import os

from cider import Cider

# read run.py results
with open('save/run.pkl', 'rb') as f:
    # direction -> {hp_i -> {img_id -> res}}
    res = pickle.load(f)

dire = 'shrunk-5/'

# read captions

cap_jsonl = pd.read_json('captions.jsonl', lines=True)
def get_captions(img_id, lang):
    return cap_jsonl[cap_jsonl['image/key'] == img_id][lang].iloc[0]['caption']

tgt_langs = ['en', 'zh']
ref_caps = {l: {} for l in tgt_langs} # tgt_lang -> {img_id -> cap}
for fname in os.listdir(dire):
    img_id = fname[:-4]
    for tgt_lang in tgt_langs:
        ref_caps[tgt_lang][img_id] = get_captions(img_id, lang)

cider = Cider()

# evaluate
for dirn in dirs:
    src_lang, tgt_lang = dirn
    for hp_i in res:
        score, scores = cider.compute_score(res[hp_i], ref_caps[dirn][hp_i])
        print('direction', dirn, 'hp_i', hp_i, 'score', score)
