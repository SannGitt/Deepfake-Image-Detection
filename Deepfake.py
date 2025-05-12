import os
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageChops, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

# Set random seed for reproducibility
np.random.seed(2)

# Set Seaborn style
sns.set(style='white', context='notebook', palette='deep')

def get_imlist(path):
    """Return a list of .jpg and .png image file paths in the given directory."""
    return [os.path.join(path, f) for f in os.listdir(path) 
            if f.endswith('.jpg') or f.endswith('.png')]

def convert_to_ela_image(path, quality):
    """Convert image to Error Level Analysis (ELA) image."""
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    ELA_filename = filename.split('.')[0] + '.ela.png'

    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)

    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    return ela_im

def prepare_image(image_path, image_size=(128, 128)):
    """Prepare and normalize image for ELA."""
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """Plot the confusion matrix."""
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    # Paths to real and fake images
    real_image_path = '/content/drive/MyDrive/Deepfake/casia/CASIA1/Au/Au_ani_0001.jpg'
    fake_image_path = '/content/drive/MyDrive/Deepfake/casia/CASIA2/Tp/Tp_D_NRN_S_N_ani10171_ani00001_12458.jpg'

    # Load and visualize real and fake images
    Image.open(real_image_path)
    convert_to_ela_image(real_image_path, 90)
    Image.open(fake_image_path)
    convert_to_ela_image(fake_image_path, 90)

    # Prepare dataset
    X, Y = [], []
    
    # Load real images
    path_real = '/content/drive/MyDrive/Deepfake/casia/CASIA2/Au'
    for dirname, _, filenames in os.walk(path_real):
        for filename in filenames:
            if filename.endswith('jpg') or filename.endswith('png'):
                full_path = os.path.join(dirname, filename)
                X.append(prepare_image(full_path))
                Y.append(1)
                if len(Y) % 500 == 0:
                    print(f'Processing {len(Y)} images')

    # Limit dataset size for real images
    random.shuffle(X)
    X = X[:2100]
    Y = Y[:2100]

    # Load fake images
    path_fake = '/content/drive/MyDrive/Deepfake/casia/CASIA2/Tp'
    for dirname, _, filenames in os.walk(path_fake):
        for filename in filenames:
            if filename.endswith('jpg') or filename.endswith('png'):
                full_path = os.path.join(dirname, filename)
                X.append(prepare_image(full_path))
                Y.append(0)
                if len(Y) % 500 == 0:
                    print(f'Processing {len(Y)} images')

    # Convert lists to arrays
    X = np.array(X)
    Y = to_categorical(Y, 2)
    
    # Reshape data
    X = X.reshape(-1, 128, 128, 3)

    # Split data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=5)

    # Define the model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(128, 128, 3)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # Model summary
    model.summary()

    # Compile the model
    optimizer = RMSprop(learning_rate=0.0005, rho=0.9, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=2, verbose=0, mode='max')
    
    # Train the model
    epochs = 30
    batch_size = 100
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val), verbose=2, callbacks=[early_stopping])

    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="Validation loss")
    ax[0].legend(loc='best', shadow=True)
    
    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1]