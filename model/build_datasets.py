#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import os
import sys
import hashlib
import shutil

from classes.Sampler import *

argv = sys.argv[1:]

if len(argv) < 1:
    print("Error: not enough argument supplied:")
    print("build_datasets.py <input path> <distribution (default: 6)>")
    exit(0)
else:
    # รับ input_path, distribution จาก argument ของ command line
    # input_path คือ path ของ dataset
    input_path = argv[0]

# distribution คือ ตัวกำหนดอัตราส่วนของ training set กับ evaluation set (default เป็น 6)
distribution = 6 if len(argv) < 2 else argv[1]

TRAINING_SET_NAME = "training_set"
EVALUATION_SET_NAME = "eval_set"

paths = []
# เก็บ path + filename (ไม่เก็นนามสกุลไฟล์) ไว้ในตัวแปร paths
# โดยมีเงื่อนไขว่า filename นั้นต้องมีไฟล์ที่นามสกุลที่เป็น .gui และ .png
for f in os.listdir(input_path):
    if f.find(".gui") != -1:
        path_gui = "{}/{}".format(input_path, f)
        file_name = f[:f.find(".gui")]

        if os.path.isfile("{}/{}.png".format(input_path, file_name)):
            path_img = "{}/{}.png".format(input_path, file_name)
            paths.append(file_name)

# กำหนด size ของ training set และ evaluation set
evaluation_samples_number = len(paths) / (distribution + 1)
training_samples_number = evaluation_samples_number * distribution

assert training_samples_number + evaluation_samples_number == len(paths)

print("Splitting datasets, training samples: {}, evaluation samples: {}".format(training_samples_number, evaluation_samples_number))

# สลับ path แบบ random
np.random.shuffle(paths)

eval_set = []
train_set = []

hashes = []
# เริ่มทำการแบ่ง training set และ evaluation set 
# ในทั้ง 2 set จะเก็บข้อมูลที่เป็น path ของ file เท่านั้น
for path in paths:
    # เลือกอ่าน flie ที่เป็น .gui
    if sys.version_info >= (3,):
        f = open("{}/{}.gui".format(input_path, path), 'r', encoding='utf-8')
    else:
        f = open("{}/{}.gui".format(input_path, path), 'r')

    with f:
        chars = ""
        for line in f:
            chars += line
        content_hash = chars.replace(" ", "").replace("\n", "")
        # เอา content file .gui มา hash ด้วย SHA256
        content_hash = hashlib.sha256(content_hash.encode('utf-8')).hexdigest()

        # ถ้า evaluation set ถูกแบ่งได้ "ครบ" ตามขนาดที่กำหนด
        # ไฟล์ที่เหลือก็จะถูกจัดไว้ใน training set
        if len(eval_set) == evaluation_samples_number:
            train_set.append(path)
        # แต่ถ้า evaluation set "ยังไม่ครบ"
        else:
            is_unique = True
            for h in hashes:
                if h is content_hash:
                    is_unique = False
                    break

            # ถ้า content ของ file ปัจจุบัน "ไม่ซ้ำ" กับ content ของ file ก่อนหน้าที่เคยเก็บไปแล้ว
            # ไฟล์ก็จะถูกจัดจัดไว้ใน evaluation set
            # เพราะต้องการให้ evaluation set มีการกระจายตัวมากที่สุด
            if is_unique:
                eval_set.append(path)
            # ถ้า content ของ file ปัจจุบัน "ซ้ำ" กับ content ของ file ก่อนหน้า
            # ไฟล์ก็จะถูกจัดจัดไว้ใน training set
            else:
                train_set.append(path)

        hashes.append(content_hash)

assert len(eval_set) == evaluation_samples_number
assert len(train_set) == training_samples_number

# สร้าง floder ของ evaluation set
if not os.path.exists("{}/{}".format(os.path.dirname(input_path), EVALUATION_SET_NAME)):
    os.makedirs("{}/{}".format(os.path.dirname(input_path), EVALUATION_SET_NAME))

# สร้าง floder ของ training set
if not os.path.exists("{}/{}".format(os.path.dirname(input_path), TRAINING_SET_NAME)):
    os.makedirs("{}/{}".format(os.path.dirname(input_path), TRAINING_SET_NAME))

# copy file นามสกุล .png และ .gui ที่ถูกแบ่งไว้เป็น evaluation set ไปไว้ใน folder ใหม่
for path in eval_set:
    shutil.copyfile("{}/{}.png".format(input_path, path), "{}/{}/{}.png".format(os.path.dirname(input_path), EVALUATION_SET_NAME, path))
    shutil.copyfile("{}/{}.gui".format(input_path, path), "{}/{}/{}.gui".format(os.path.dirname(input_path), EVALUATION_SET_NAME, path))

# copy file นามสกุล .png และ .gui ที่ถูกแบ่งไว้เป็น training set ไปไว้ใน folder ใหม่
for path in train_set:
    shutil.copyfile("{}/{}.png".format(input_path, path), "{}/{}/{}.png".format(os.path.dirname(input_path), TRAINING_SET_NAME, path))
    shutil.copyfile("{}/{}.gui".format(input_path, path), "{}/{}/{}.gui".format(os.path.dirname(input_path), TRAINING_SET_NAME, path))

print("Training dataset: {}/training_set".format(os.path.dirname(input_path), path))
print("Evaluation dataset: {}/eval_set".format(os.path.dirname(input_path), path))
