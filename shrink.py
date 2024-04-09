from PIL import Image
import sys
import os

inp_dbase = 'images/'
res_dbase = 'sample_imgs_3/'
if not os.path.exists(res_dbase):
    os.mkdir(res_dbase)

samp_i, n_samp = 0, 5
shrink_factor = 5
for fname in os.listdir(inp_dbase):
    img = Image.open(inp_dbase + fname)
    img.save(res_dbase + fname)

    height, width = img.size
    shrunk_img = img.resize((height // shrink_factor, width // shrink_factor))
    shrunk_img.save(res_dbase + os.path.split(fname)[1][:-4] + '_shrunk.jpg')

    samp_i += 1
    if samp_i == n_samp:
        break
