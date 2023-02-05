import h5py
import numpy as np

def load_dataset():

    # You can prepare your h5 dataset of the same format
    # Include the 'train_set_x', 'train_set_y', 'list_classes' 
    # and 'true_labels' dirs

    f = h5py.File('family_dataset.h5', 'r')
    train_set_x_orig = np.array(f["train_set_x"][:])
    train_set_y_orig = np.array(f["train_set_y"][:])
    classes = np.array(f["list_classes"])
    
    labels = np.array(f["true_labels"])

    return train_set_x_orig, train_set_y_orig, classes, labels


train_set_x, train_set_y, classes, labels = load_dataset()

