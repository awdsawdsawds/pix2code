#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import os
import sys
import pickle
from datetime import datetime

from classes.dataset.Generator import *
from classes.model.pix2code import *

HISTORY_DIR = "../history/"


def save_history(filename, history):
    print("save history to file")
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)

    now = datetime.now().strftime('%d:%m:%Y,%H:%M:%S')

    with open(f"{HISTORY_DIR}/{filename}-{now}.pickle","wb") as pickle_out:
        pickle.dump(history, pickle_out)

def run(input_path, eval_input_path, output_path, is_memory_intensive=False, pretrained_model=None):
    np.random.seed(1234)

    dataset = Dataset()
    # TODO: ย่อ comment
    # load training data เพื่อเอามาสร้างตัวแปรสำหรับเข้าโมเดล ซึ่งจะมี
    # 1. dataset.ids คือ unigue index โดยสร้างจากชื่อไฟล์ใน training set
    # 2. dataset.size คือ ตัวเลขจำนวนของ training set
    # 3. dataset.partial_sequences เป็น list ข้อมูล จาก GUI ที่เก็บเป็น partial sequences (ข้อมูลตั้งแต่เริ่มต้นถึงปัจจุบัน)
    #   มี 2 แบบซึ่งในที่นี้เราจะใช้ format one hot endcoding
    #   - ถ้าเป็น generate_binary_sequences=True จะ format one hot vector
    #      : [[[1, 0, ..., 0]], [[1, 0, ..., 0], [0, 1, ..., 0]], ..., [[1, 0, ..., 0], [0, 1, ..., 0], ..., [0, 0, ..., 1]]]
    #   - ถ้าเป็น generate_binary_sequences=Fasle จะ format by index
    #      : [[0], [0, 1], ..., [0, 1, ..., n]]
    # 4. dataset.next_words คือ list ของ label ที่เป็น one hot vector: [[1, 0, 0], ..., [0, 0, 1]]
    # 5. dataset.voc.vocabulary เป็น dict ของ คำที่พบใน .gui
    #   : { "<START>": 0, "head": 1, "{": 2, ..., "<END>": n }
    # 6. dataset.voc.token_lookup เป็น dict ที่คล้ายกับ dataset.voc.vocabulary แต่จะสลับตรง key และ value
    #   : { "0": "<START>", "1": "head", "2": "{", ..., n: "<END>" }
    # 7. dataset.voc.binary_vocabulary เป็น dict ที่เกิดจากการแปลง dataset.voc.vocabulary เอามาทำ one hot vector
    #   : { "<START>": [1, 0, 0, ..., 0], "head": [0, 1, 0, ..., 0], "{": [0, 0, 1, ..., 0], ..., "<END>": [0, 0, 0, ..., 1] }
    # 8. dataset.voc.size คือ ตัวเลขจำนวนของ vocabulary: ตัวเลขจำนวนของคำที่พบใน .gui
    dataset.load_train_and_eval(input_path, generate_binary_sequences=True)
    # save input_shape, output_size, size ในไฟล์ meta_dataset.npy
    dataset.save_metadata(output_path)
    # save dataset.voc.binary_vocabulary metadata ในไฟล์ words.vocab
    dataset.voc.save(output_path)

    # ถ้า is_memory_intensive = False เราจะ get metadata จาก ตัวแปร dataset ตรงๆ  
    if not is_memory_intensive:
        dataset.convert_arrays()

        input_shape = dataset.input_shape
        output_size = dataset.output_size

        print(len(dataset.input_images), len(dataset.partial_sequences), len(dataset.next_words))
        print(dataset.input_images.shape, dataset.partial_sequences.shape, dataset.next_words.shape)
    # ถ้า is_memory_intensive = True เราจะสร้าง generator ขึ้นมา
    else:
        gui_paths, img_paths = Dataset.load_paths_only(input_path)

        input_shape = dataset.input_shape
        output_size = dataset.output_size
        steps_per_epoch = dataset.size / BATCH_SIZE

        voc = Vocabulary()
        voc.retrieve(output_path)

        generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE, generate_binary_sequences=True)

    model = pix2code(input_shape, output_size, output_path)

    if pretrained_model is not None:
        model.model.load_weights(pretrained_model)

    eval = ([dataset.eval_input_images, dataset.eval_partial_sequences], dataset.eval_next_words)
    if not is_memory_intensive:
        history = model.fit(dataset.input_images, dataset.partial_sequences, dataset.next_words, eval)
        save_history('history-feed-all', history)
    else:
        history = model.fit_generator(generator, eval, steps_per_epoch=steps_per_epoch)
        save_history('history-feed-stream', history)

if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Error: not enough argument supplied:")
        print("train.py <input path> <output path> <is memory intensive (default: 0)> <pretrained weights (optional)>")
        exit(0)
    else:
        # รับ input_path, output_path, use_generator, pretrained_weigths จาก argument ของ command line
        # input_path คือ path ของ dataset ควรเป็น path จาก step convert_imgs_to_arrays
        input_path = argv[0]

        eval_input_path = argv[1]
        # output_path คือ path ที่จะ save output ของ step นี้
        output_path = argv[2]
        # use_generator คือ สิ่งที่บอกว่าเราจะใช้ option memory intensive รึเปล่า (default เป็น False)
        use_generator = False if len(argv) < 4 else True if int(argv[3]) == 1 else False
        # pretrained_weigths คือ path ของ pretrained weigths ต้องการใส่ไปใน model (default เป็น None)
        pretrained_weigths = None if len(argv) < 5 else argv[4]

    print("=======")
    print("input_path", input_path)
    print("eval_input_path", eval_input_path)
    print("output_path", output_path)
    print("use_generator", use_generator)
    print("pretrained_weigths", pretrained_weigths)
    print("=======")
    run(input_path, eval_input_path, output_path, is_memory_intensive=use_generator, pretrained_model=pretrained_weigths)
