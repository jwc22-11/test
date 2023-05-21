# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import random

print(tf.__version__, tf.config.list_physical_devices('GPU'))

SR = 16000
target_time = 5
audio_length = int(SR * target_time)

# %%
interpreter = tf.lite.Interpreter(model_path='/home/jwchoi/gradu/tflite/f_4M.tflite')
interpreter.allocate_tensors()

interpreter_fp16 = tf.lite.Interpreter(model_path='/home/jwchoi/gradu/tflite/f_4M_f16.tflite')
interpreter_fp16.allocate_tensors()

# %%
x = np.load('/home/jwchoi/gradu/tflite/testdata_5s.npy')

# %%
import time

test_image = np.expand_dims(x, axis=0).astype(np.float32)

a = time.time()
input_index = interpreter_fp16.get_input_details()[0]["index"]
output_index = interpreter_fp16.get_output_details()[0]["index"]

b = time.time()
print((b-a)*1000.0, 'ms')
interpreter_fp16.set_tensor(input_index, test_image)
interpreter_fp16.invoke()
c = time.time()
print((c-b)*1000.0, 'ms')

predictions = interpreter_fp16.get_tensor(output_index)

print(np.argmax(predictions))


