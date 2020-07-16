#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import os
import sys
import shutil

from classes.Utils import *
from classes.model.Config import *

argv = sys.argv[1:]

if len(argv) < 2:
    print("Error: not enough argument supplied:")
    print("convert_imgs_to_arrays.py <input path> <output path>")
    exit(0)
else:
    # รับ input_path, output_path จาก argument ของ command line
    # input_path คือ path ของ dataset ควรเป็น path จาก step build_dataset ที่เป็น training set floder
    input_path = argv[0]
    # output_path คือ path ที่จะ save output ของ step นี้
    output_path = argv[1]

if not os.path.exists(output_path):
    os.makedirs(output_path)

print("Converting images to numpy arrays...")

# ทำการ copy ไฟล์ทั้งหมด floder ที่ระบุไว้
for f in os.listdir(input_path):
    if f.find(".png") != -1:
        # resize แบบ normalize รูป .png แล้ว return array กลับมา
        img = Utils.get_preprocessed_img("{}/{}".format(input_path, f), IMAGE_SIZE)
        file_name = f[:f.find(".png")]

        # นำ array ดังกล่าว ไป save เป็นไฟล์ .npz (array file with compress)
        # จากนั้นจะ save output ไป floder ใหม่
        np.savez_compressed("{}/{}".format(output_path, file_name), features=img)
        retrieve = np.load("{}/{}.npz".format(output_path, file_name))["features"]

        assert np.array_equal(img, retrieve)

        # สำหรับไฟล์ .gui copy ไป floder ใหม่ได้ทันที
        shutil.copyfile("{}/{}.gui".format(input_path, file_name), "{}/{}.gui".format(output_path, file_name))

print("Numpy arrays saved in {}".format(output_path))
