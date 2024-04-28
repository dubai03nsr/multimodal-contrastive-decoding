# /home1/09896/victorwang37/.cache/huggingface/modules/transformers_modules/openbmb/MiniCPM-V/bec7d1cd1c9e804c064ec291163e40624825eaaa/
# 9d9376bea053209273767588969282a9f3ef95c0/
# /data/vwang/.cache/huggingface/modules/transformers_modules/openbmb/MiniCPM-V/496723fac5bb5e1c31b28faec94669f249eb8c54
# /data/vwang/.cache/huggingface/modules/transformers_modules/openbmb/MiniCPM-V/a5833d2a6ae01c3f07c6b3c5c12ceb9c7a9791f0

from eval import eval_cider, eval_comet, init_comet

# init_comet()
for l in [None, 'en', 'zh']:
    print(eval_cider('run5', 2, lang_filter=l))