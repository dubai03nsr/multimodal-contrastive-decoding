import pickle
import os
import pandas as pd
from comet import download_model, load_from_checkpoint

from cider import Cider

cap_jsonl = pd.read_json('captions.jsonl', lines=True)
def get_captions(img_id, lang):
    return cap_jsonl[cap_jsonl['image/key'] == img_id][lang].iloc[0]['caption']
def get_caption(img_id, lang, idx=0):
    return cap_jsonl[cap_jsonl['image/key'] == img_id][lang].iloc[0]['caption'][idx]
def get_img_lang(img_id):
    return cap_jsonl[cap_jsonl['image/key'] == img_id]['image/locale'].iloc[0]

def eval_cider(fname, n_hp, lang_filter=None):
    # read run.py results
    with open('save/' + fname + '.pkl', 'rb') as f:
        # direction -> {hp_i -> {img_id -> res}}
        res = pickle.load(f)

    dirs = [('en', 'zh'), ('zh', 'en')]
    ciders = {d[1]: Cider(lang=d[1]) for d in dirs}
    scores = {d: {} for d in dirs} # direction -> {hp_i -> score}
    if lang_filter is None:
        filtered_img_ids = list(res[dirs[0]][0].keys())
    else:
        filtered_img_ids = [img_id for img_id in res[dirs[0]][0] if get_img_lang(img_id) == lang_filter]
    print(len(filtered_img_ids), 'imgs')

    tgt_langs = ['en', 'zh']
    ref_caps = {l: {} for l in tgt_langs} # tgt_lang -> {img_id -> captions}
    for img_id in filtered_img_ids:
        for tgt_lang in tgt_langs:
            ref_caps[tgt_lang][img_id] = get_captions(img_id, tgt_lang)

    # evaluate
    for dirn in dirs:
        src_lang, tgt_lang = dirn
        # for hp_i in res[dirn]:
        for hp_i in range(n_hp):
            # filter res by filtered_img_ids
            res[dirn][hp_i] = {img_id: res[dirn][hp_i][img_id] for img_id in filtered_img_ids}

            score, returned_scores = ciders[tgt_lang].compute_score(ref_caps[tgt_lang], res[dirn][hp_i])
            scores[dirn][hp_i] = score
            # print('direction', dirn, 'hp_i', hp_i, 'score', score)

    # with open('save/scores3.pkl', 'wb') as f:
    #     pickle.dump(scores, f)

    return scores

def init_comet():
    global model
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

def eval_comet(fname, n_hp, lang_filter=None):
    # read run.py results
    with open('save/' + fname + '.pkl', 'rb') as f:
        # direction -> {hp_i -> {img_id -> res}}
        res = pickle.load(f)

    global model
    # load model
    # model_path = download_model("Unbabel/wmt22-comet-da")
    # model = load_from_checkpoint(model_path)

    dirs = [('en', 'zh'), ('zh', 'en')]
    scores = {d: {} for d in dirs} # direction -> {hp_i -> score}
    if lang_filter is None:
        filtered_img_ids = list(res[dirs[0]][0].keys())
    else:
        filtered_img_ids = [img_id for img_id in res[dirs[0]][0] if get_img_lang(img_id) == lang_filter]
    print(len(filtered_img_ids), 'imgs')

    caps = {d: [[] for _ in range(n_hp)] for d in dirs} # dir -> [hp_i -> [{src, ref, hyp}]]
    for img_id in filtered_img_ids:
        for d in dirs:
            src_lang, tgt_lang = d
            src_cap = get_caption(img_id, src_lang)
            tgt_cap = get_caption(img_id, tgt_lang)
            for hp_i in range(n_hp):
                caps[d][hp_i].append({
                    'src': src_cap,
                    'ref': tgt_cap,
                    'mt': res[d][hp_i][img_id]
                })

    # evaluate
    for d in dirs:
        src_lang, tgt_lang = d
        # for hp_i in res[dirn]:
        for hp_i in range(n_hp):
            scores[d][hp_i] = model.predict(caps[d][hp_i], batch_size=8, gpus=1).system_score
            print('direction', d, 'hp_i', hp_i, 'score', scores[d][hp_i])

    # with open('save/scores3.pkl', 'wb') as f:
    #     pickle.dump(scores, f)

    return scores