# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import random
import time
import os

print(tf.__version__)

# %%
f_set = []

f_4M = tf.lite.Interpreter(model_path= './tflite/f_4M.tflite')
f_4M.allocate_tensors()
f_set.append(f_4M)

f_4M_fp16 = tf.lite.Interpreter(model_path='./tflite/f_4M_f16.tflite')
f_4M_fp16.allocate_tensors()
f_set.append(f_4M_fp16)

f_1M = tf.lite.Interpreter(model_path='./tflite/f_1M.tflite')
f_1M.allocate_tensors()
f_set.append(f_1M)

f_1M_fp16 = tf.lite.Interpreter(model_path='./tflite/f_1M_f16.tflite')
f_1M_fp16.allocate_tensors()
f_set.append(f_1M_fp16)

f_410k = tf.lite.Interpreter(model_path='./tflite/f_410k.tflite')
f_410k.allocate_tensors()
f_set.append(f_410k)

f_410k_fp16 = tf.lite.Interpreter(model_path='./tflite/f_410k_f16.tflite')
f_410k_fp16.allocate_tensors()
f_set.append(f_410k_fp16)

f_181k = tf.lite.Interpreter(model_path='./tflite/f_181k.tflite')
f_181k.allocate_tensors()
f_set.append(f_181k)

f_181k_fp16 = tf.lite.Interpreter(model_path='./tflite/f_181k_f16.tflite')
f_181k_fp16.allocate_tensors()
f_set.append(f_181k_fp16)

f_120k = tf.lite.Interpreter(model_path='./tflite/f_120k.tflite')
f_120k.allocate_tensors()
f_set.append(f_120k)

f_120k_fp16 = tf.lite.Interpreter(model_path='./tflite/f_120k_f16.tflite')
f_120k_fp16.allocate_tensors()
f_set.append(f_120k_fp16)

f_70k = tf.lite.Interpreter(model_path='./tflite/f_70k.tflite')
f_70k.allocate_tensors()
f_set.append(f_70k)

f_70k_fp16 = tf.lite.Interpreter(model_path='./tflite/f_70k_f16.tflite')
f_70k_fp16.allocate_tensors()
f_set.append(f_70k_fp16)

f_29k = tf.lite.Interpreter(model_path='./tflite/f_29k.tflite')
f_29k.allocate_tensors()
f_set.append(f_29k)

f_29k_fp16 = tf.lite.Interpreter(model_path='./tflite/f_29k_f16.tflite')
f_29k_fp16.allocate_tensors()
f_set.append(f_29k_fp16)

print('complite')

# %%
TEST_PATH = './nparray_5s'

TEST = sorted(os.listdir(TEST_PATH))

TEST_fnames = []
TEST_datalen = 0

for i in range(len(TEST)):
    TEST_fnames.append(np.array(os.listdir(os.path.join(TEST_PATH, TEST[i]))))
    TEST_datalen += len(TEST_fnames[i])
    print(TEST[i], TEST_fnames[i].shape)
print(TEST_datalen)

TEST_filenames = []
TEST_targets = []

for i in range(len(TEST)):
    for j in range(len(TEST_fnames[i])):
        TEST_filenames.append(os.path.join(TEST_PATH, TEST[i], TEST_fnames[i][j]))
        TEST_targets.append(i)

# %%
sum = 0

for i in f_set:
    for j in TEST_filenames:
        f = np.load(j)
        test_5s = np.expand_dims(f, axis=0).astype(np.float32)

        input_index = i.get_input_details()[0]["index"]
        output_index = i.get_output_details()[0]["index"]

        a = time.time()
        i.set_tensor(input_index, test_5s)
        i.invoke()
        b = time.time() - a
        sum += b
        predictions = i.get_tensor(output_index)
        # print(np.argmax(predictions))
    print(sum / 2.1, 'ms')
    sum = 0


