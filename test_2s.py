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
t_set = []

t_4M = tf.lite.Interpreter(model_path='t_4M.tflite')
t_4M.allocate_tensors()
t_set.append(t_4M)

t_4M_fp16 = tf.lite.Interpreter(model_path='t_4M_f16.tflite')
t_4M_fp16.allocate_tensors()
t_set.append(t_4M_fp16)

t_1M = tf.lite.Interpreter(model_path='t_1M.tflite')
t_1M.allocate_tensors()
t_set.append(t_1M)

t_1M_fp16 = tf.lite.Interpreter(model_path='t_1M_f16.tflite')
t_1M_fp16.allocate_tensors()
t_set.append(t_1M_fp16)

t_410k = tf.lite.Interpreter(model_path='t_410k.tflite')
t_410k.allocate_tensors()
t_set.append(t_410k)

t_410k_fp16 = tf.lite.Interpreter(model_path='t_410k_f16.tflite')
t_410k_fp16.allocate_tensors()
t_set.append(t_410k_fp16)

t_181k = tf.lite.Interpreter(model_path='t_181k.tflite')
t_181k.allocate_tensors()
t_set.append(t_181k)

t_181k_fp16 = tf.lite.Interpreter(model_path='t_181k_f16.tflite')
t_181k_fp16.allocate_tensors()
t_set.append(t_181k_fp16)

t_120k = tf.lite.Interpreter(model_path='t_120k.tflite')
t_120k.allocate_tensors()
t_set.append(t_120k)

t_120k_fp16 = tf.lite.Interpreter(model_path='t_120k_f16.tflite')
t_120k_fp16.allocate_tensors()
t_set.append(t_120k_fp16)

t_70k = tf.lite.Interpreter(model_path='t_70k.tflite')
t_70k.allocate_tensors()
t_set.append(t_70k)

t_70k_fp16 = tf.lite.Interpreter(model_path='t_70k_f16.tflite')
t_70k_fp16.allocate_tensors()
t_set.append(t_70k_fp16)

t_29k = tf.lite.Interpreter(model_path='t_29k.tflite')
t_29k.allocate_tensors()
t_set.append(t_29k)

t_29k_fp16 = tf.lite.Interpreter(model_path='t_29k_f16.tflite')
t_29k_fp16.allocate_tensors()
t_set.append(t_29k_fp16)

print('complite')

# %%
TEST_PATH = './nparray_2s'

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

for i in t_set:
    for j in TEST_filenames:
        t = np.load(j)
        test_5s = np.expand_dims(t, axis=0).astype(np.float32)

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


