# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import random
import time

print(tf.__version__)

# %%
f_set = []
base_path = '/home/jwchoi/gradu/tflite/'

f_4M = tf.lite.Interpreter(model_path= base_path + 'f_4M.tflite')
f_4M.allocate_tensors()
f_set.append(f_4M)

f_4M_fp16 = tf.lite.Interpreter(model_path= base_path + 'f_4M_f16.tflite')
f_4M_fp16.allocate_tensors()
f_set.append(f_4M_fp16)

f_1M = tf.lite.Interpreter(model_path= base_path + 'f_1M.tflite')
f_1M.allocate_tensors()
f_set.append(f_1M)

f_1M_fp16 = tf.lite.Interpreter(model_path= base_path + 'f_1M_f16.tflite')
f_1M_fp16.allocate_tensors()
f_set.append(f_1M_fp16)

f_410k = tf.lite.Interpreter(model_path= base_path + 'f_410k.tflite')
f_410k.allocate_tensors()
f_set.append(f_410k)

f_410k_fp16 = tf.lite.Interpreter(model_path= base_path + 'f_410k_f16.tflite')
f_410k_fp16.allocate_tensors()
f_set.append(f_410k_fp16)

f_181k = tf.lite.Interpreter(model_path= base_path + 'f_181k.tflite')
f_181k.allocate_tensors()
f_set.append(f_181k)

f_181k_fp16 = tf.lite.Interpreter(model_path= base_path + 'f_181k_f16.tflite')
f_181k_fp16.allocate_tensors()
f_set.append(f_181k_fp16)

f_120k = tf.lite.Interpreter(model_path= base_path + 'f_120k.tflite')
f_120k.allocate_tensors()
f_set.append(f_120k)

f_120k_fp16 = tf.lite.Interpreter(model_path= base_path + 'f_120k_f16.tflite')
f_120k_fp16.allocate_tensors()
f_set.append(f_120k_fp16)

f_70k = tf.lite.Interpreter(model_path= base_path + 'f_70k.tflite')
f_70k.allocate_tensors()
f_set.append(f_70k)

f_70k_fp16 = tf.lite.Interpreter(model_path= base_path + 'f_70k_f16.tflite')
f_70k_fp16.allocate_tensors()
f_set.append(f_70k_fp16)

f_29k = tf.lite.Interpreter(model_path= base_path + 'f_29k.tflite')
f_29k.allocate_tensors()
f_set.append(f_29k)

f_29k_fp16 = tf.lite.Interpreter(model_path= base_path + 'f_29k_f16.tflite')
f_29k_fp16.allocate_tensors()
f_set.append(f_29k_fp16)

f = np.load('/home/jwchoi/gradu/tflite/testdata_5s.npy')

print('complite')

# %%
test_5s = np.expand_dims(f, axis=0).astype(np.float32)

for i in f_set:
    input_index = i.get_input_details()[0]["index"]
    output_index = i.get_output_details()[0]["index"]

    a = time.time()
    i.set_tensor(input_index, test_5s)
    i.invoke()
    b = time.time()
    print((b-a)*1000.0, 'ms')

    predictions = i.get_tensor(output_index)

    # print(np.argmax(predictions))

# %%
t_set = []

t_4M = tf.lite.Interpreter(model_path= base_path + 't_4M.tflite')
t_4M.allocate_tensors()
t_set.append(t_4M)

t_4M_fp16 = tf.lite.Interpreter(model_path= base_path + 't_4M_f16.tflite')
t_4M_fp16.allocate_tensors()
t_set.append(t_4M_fp16)

t_1M = tf.lite.Interpreter(model_path= base_path + 't_1M.tflite')
t_1M.allocate_tensors()
t_set.append(t_1M)

t_1M_fp16 = tf.lite.Interpreter(model_path= base_path + 't_1M_f16.tflite')
t_1M_fp16.allocate_tensors()
t_set.append(t_1M_fp16)

t_410k = tf.lite.Interpreter(model_path= base_path + 't_410k.tflite')
t_410k.allocate_tensors()
t_set.append(t_410k)

t_410k_fp16 = tf.lite.Interpreter(model_path= base_path + 't_410k_f16.tflite')
t_410k_fp16.allocate_tensors()
t_set.append(t_410k_fp16)

t_181k = tf.lite.Interpreter(model_path= base_path + 't_181k.tflite')
t_181k.allocate_tensors()
t_set.append(t_181k)

t_181k_fp16 = tf.lite.Interpreter(model_path= base_path + 't_181k_f16.tflite')
t_181k_fp16.allocate_tensors()
t_set.append(t_181k_fp16)

t_120k = tf.lite.Interpreter(model_path= base_path + 't_120k.tflite')
t_120k.allocate_tensors()
t_set.append(t_120k)

t_120k_fp16 = tf.lite.Interpreter(model_path= base_path + 't_120k_f16.tflite')
t_120k_fp16.allocate_tensors()
t_set.append(t_120k_fp16)

t_70k = tf.lite.Interpreter(model_path= base_path + 't_70k.tflite')
t_70k.allocate_tensors()
t_set.append(t_70k)

t_70k_fp16 = tf.lite.Interpreter(model_path= base_path + 't_70k_f16.tflite')
t_70k_fp16.allocate_tensors()
t_set.append(t_70k_fp16)

t_29k = tf.lite.Interpreter(model_path= base_path + 't_29k.tflite')
t_29k.allocate_tensors()
t_set.append(t_29k)

t_29k_fp16 = tf.lite.Interpreter(model_path= base_path + 't_29k_f16.tflite')
t_29k_fp16.allocate_tensors()
t_set.append(t_29k_fp16)

t = np.load('/home/jwchoi/gradu/tflite/testdata_2s.npy')

print('complite')

# %%
test_2s = np.expand_dims(t, axis=0).astype(np.float32)

for i in t_set:
    input_index = i.get_input_details()[0]["index"]
    output_index = i.get_output_details()[0]["index"]

    a = time.time()
    i.set_tensor(input_index, test_2s)
    i.invoke()
    b = time.time()
    print((b-a)*1000.0, 'ms')

    predictions = i.get_tensor(output_index)

    # print(np.argmax(predictions))


