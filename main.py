import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_unet.models import custom_unet
from keras_unet.utils import plot_segm_history
import tensorflow_datasets as tfds
from keras.utils import to_categorical


"""
побудувати сегментатор. датасет довільний. 
так як задача обчислювально складніша, ніж була до цього - можна вчити невелику кількість епох
головне розібратися в даних, в постановці задачі, і побудувати правильний флоу.
"""

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(
        datapoint['segmentation_mask'],
        (128, 128),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


# converting dataset with method map
train_dataset = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# converting dataset into list
train_data = list(train_dataset)
test_data = list(test_dataset)

# creating array X and y
X_train = np.array([item[0] for item in train_data])
y_train = np.array([item[1] for item in train_data])

X_test = np.array([item[0] for item in test_data])
y_test = np.array([item[1] for item in test_data])

y_train[y_train == 2] = 1
y_test[y_test == 2] = 1

unique_values = np.unique(y_train)
print("uniq value in mask:", unique_values)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# first five mask from train
num_masks_to_display = 5
plt.figure(figsize=(15, 5))
for i in range(num_masks_to_display):
    plt.subplot(1, num_masks_to_display, i + 1)
    plt.imshow(y_train[i][:, :, 0], cmap='gray')
    plt.title('mask {}'.format(i + 1))
plt.show()

model = custom_unet(
    input_shape=(128, 128, 3),
    use_batch_norm=False,
    num_classes=1,
    filters=64,
    dropout=0.2,
    output_activation='sigmoid')

optimizer = keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[keras.metrics.IoU(num_classes=2, target_class_ids=[0])])

# train model
history = model.fit(X_train, y_train, epochs=2, validation_split=0.2)

test_loss, test_mean_iou = model.evaluate(X_test, y_test)

print("loss:", test_loss)
print("mean IoU:", test_mean_iou)