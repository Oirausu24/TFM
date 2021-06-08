from glob import glob
import matplotlib.pyplot as plt

from keras.applications import VGG16

from keras import models
from keras import layers
from keras import optimizers

# variables
path_to_images = 'PATH'
batch_size = 32
class_indices = {'red': 0, 'green': 1, 'black': 2}

image_files = glob(path_to_images + "/*.tiff")

conv_base = VGG16(weights=None,
include_top=False,
input_shape=(128, 128, 37))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['acc','mse'])


import random

green = image_files[:len(image_files)//3]
red = image_files[204:408]
black = image_files[408:]

random.seed(12)
random.shuffle(green)
random.shuffle(red)
random.shuffle(black)

train_data = green[:123]+red[:123]+black[:123]
validation_data = green[123:164]+red[123:164]+black[123:164]
test_data = green[163:]+red[163:]+black[163:]

test_train_data = green[:123]+red[:123]+black[:123]+green[163:]+red[163:]+black[163:]

random.seed(13)
random.shuffle(train_data)
random.shuffle(validation_data)
random.shuffle(test_data)
random.shuffle(test_train_data)

no_augmentation_parameters = {'flip': False,
                            'zoom': 1.0,
                            'shift': 0.0,
                            'rotation': 0.0,
                            'sheer': 0.0,
                            'noising': None}
augmentation_parameters = {'flip': True,
                            'zoom': 1.2,
                            'shift': 0.15,
                            'rotation': 20.0,
                            'sheer': 0.015,
                            'noising': None}

validation_generator = hyperspectral_image_generator(validation_data, class_indices,
                                                batch_size=32,
                                                image_mean=None,
                                                rotation_range=no_augmentation_parameters['rotation'],
                                                horizontal_flip=no_augmentation_parameters['flip'],
                                                vertical_flip=no_augmentation_parameters['flip'],
                                                speckle_noise=no_augmentation_parameters['noising'],
                                                shear_range=no_augmentation_parameters['sheer'],
                                                scale_range=no_augmentation_parameters['zoom'],
                                                transform_range=no_augmentation_parameters['shift']
                                                )
train_generator = hyperspectral_image_generator(train_data, class_indices,
                                                batch_size=32,
                                                image_mean=None,
                                                rotation_range=augmentation_parameters['rotation'],
                                                horizontal_flip=augmentation_parameters['flip'],
                                                vertical_flip=augmentation_parameters['flip'],
                                                speckle_noise=augmentation_parameters['noising'],
                                                shear_range=augmentation_parameters['sheer'],
                                                scale_range=augmentation_parameters['zoom'],
                                                transform_range=augmentation_parameters['shift']
                                                )


history = model.fit(train_generator,
                    steps_per_epoch=len(train_data) // batch_size,
                    epochs=25,
                    validation_data=validation_generator,
                    validation_steps=5)


model.save('VGG16_model.h5')

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_vs_epochs.png')

plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('acc_vs_epochs.png')

model = keras.models.load_model('VGG16_model.h5')

test_generator = hyperspectral_image_generator(test_data, class_indices,
                                                batch_size=32,
                                                image_mean=None,
                                                rotation_range=no_augmentation_parameters['rotation'],
                                                horizontal_flip=no_augmentation_parameters['flip'],
                                                vertical_flip=no_augmentation_parameters['flip'],
                                                speckle_noise=no_augmentation_parameters['noising'],
                                                shear_range=no_augmentation_parameters['sheer'],
                                                scale_range=no_augmentation_parameters['zoom'],
                                                transform_range=no_augmentation_parameters['shift']
                                                )

results = model.evaluate(test_generator, steps=5)

results
