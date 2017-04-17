from keras.preprocessing.image import ImageDataGenerator , array_to_img, img_to_array, load_img
from keras.models import Sequential , model_from_json
from keras.layers import Conv2D, MaxPooling2D , Convolution2D, MaxPooling2D, ZeroPadding2D, Dropout, Flatten, Dense
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras import backend as K
import numpy as np
import csv
from scipy.misc import imresize
import os
import time
from pathlib import Path


# dimensions of our images.
img_width, img_height = 1000, 664

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_data_dir = 'data/test'
weights_path = 'QRandTarp.h5'
nb_train_samples = 158
nb_validation_samples = 158
nb_test_samples = 8
epochs = 70
batch_size = 7

np.random.seed(100)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')


def predict_labels(model):
    """writes test image labels and predictions to csv"""
    base_path = "data/test/"

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=7,
        shuffle=False,
        class_mode=None)

    with open("prediction.csv", "w") as f:
        tmps1 = time.clock()
        p_writer = csv.writer(f, delimiter=';', lineterminator='\n')
        for _, _, imgs in os.walk(base_path):
            for im in imgs:
                pic_id = im.split(".")[0]
                img = load_img(base_path + im)
                img = imresize(img, size=(img_height, img_width))
                test_x = img_to_array(img).reshape(3, img_width,  img_height)
                test_x = test_x.reshape((1,) + test_x.shape)
                test_generator = test_datagen.flow(test_x,
                                                   batch_size=1,
                                                   shuffle=False)
                prediction = model.predict_generator(test_generator, 1)[0][0]
                p_writer.writerow([pic_id, prediction])
    tmps2 = time.clock()
    print ("Exec time =", (tmps2 - tmps1), "seconds")


if __name__ == "__main__":

    my_file = Path(weights_path)
    if my_file.is_file():
        model = load_model(weights_path)
        predict_labels(model)
    else:
        tmps3 = time.clock()
        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            #zoom_range=0.2,
            horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

        model.save(weights_path)
        tmps4 = time.clock()
        print("Exec time =", (tmps4 - tmps3), "seconds")
        predict_labels(model)
