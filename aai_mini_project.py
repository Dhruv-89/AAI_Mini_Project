
from google.colab import drive
drive.mount('/content/drive')

import zipfile

# Specify the path to your zip file
zip_file_path = '/content/drive/My Drive/res (1).zip'
output_dir = '/content/drive/My Drive/'

# Unzipping the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

print("File unzipped successfully!")

import os
import pandas as pd
import keras
import cv2
import tensorflow.keras.models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

from sklearn.compose import ColumnTransformer
from keras import Sequential
from keras.src.layers import Dense, Flatten, Input
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from tensorflow.python import layers

IMG_SIZE = 224
BATCH_SIZE = 64

NUM_CLASSES = 3

inputs = Input(shape=(224, 224, 3))

base_model = keras.applications.EfficientNetB0(
    include_top=True,
    weights=None,
    classes=3
)(inputs)

model = tensorflow.keras.models.Model(inputs, base_model)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
              )

size = (IMG_SIZE, IMG_SIZE)
data = []
images = []
labels = []
for item in os.listdir("/content/drive/My Drive/res/data_resized/train"):
    for img in os.listdir(f"/content/drive/My Drive/res/data_resized/train/{item}"):
        data.append((item, f"/content/drive/My Drive/res/data_resized/train/{item}/{img}"))
        labels.append(item)
        images.append(cv2.imread(f"/content/drive/My Drive/res/data_resized/train/{item}/{img}"))

dataframe = pd.DataFrame(data=data, columns=["Labels", 'image'])

images = np.array(images)

images = images.astype("float32") / 255

print(model.summary())

y_label_enc = LabelEncoder()
y = y_label_enc.fit_transform(labels)

y = y.reshape(-1, 1)

ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')

Y = ct.fit_transform(y)

images, Y = shuffle(images, Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.05, random_state=415)
hist = model.fit(train_x, train_y, epochs=10, verbose=2)

model.save('/content/drive/My Drive/res/model.h5')

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
import os
import pandas as pd
import keras
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

from sklearn.compose import ColumnTransformer
from keras import Sequential
from keras.src.layers import Dense, Flatten, Input
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

model = tf.keras.models.load_model('/content/drive/My Drive/res/model.h5')

img = cv2.imread("/content/drive/My Drive/res/predict/aloevera_1.jpeg")
print(img.shape)
img = cv2.resize(img, (224, 224))
print(img.shape)
img = np.expand_dims(img, axis=0)
img = tf.keras.applications.imagenet_utils.preprocess_input(img)
p = model.predict(img)

labels = ['Aloevera', 'Betel', 'Neem']
print(list(p[0]))
print("Predictions: ", labels[list(p[0]).index(max(p[0]))])

import tensorflow as tf
import os
import pandas as pd
import keras
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

from sklearn.compose import ColumnTransformer
from keras import Sequential
from keras.src.layers import Dense, Flatten, Input
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

model = tf.keras.models.load_model('/content/drive/My Drive/res/model.h5')

img = cv2.imread("/content/drive/My Drive/res/predict/neem_6.png")
print(img.shape)
img = cv2.resize(img, (224, 224))
print(img.shape)
img = np.expand_dims(img, axis=0)
img = tf.keras.applications.imagenet_utils.preprocess_input(img)
p = model.predict(img)

labels = ['Aloevera', 'Betel', 'Neem']
print(list(p[0]))
print("Predictions: ", labels[list(p[0]).index(max(p[0]))])