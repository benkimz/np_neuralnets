import numpy as np
from scipy.signal import correlate
from PIL import Image
import matplotlib.pyplot as plt

def to_array_img(image):
    image = Image.open(r""+str(image))
    array = np.array(image)
    array = array.reshape(1,array.shape[0],array.shape[1],-1)
    return array

def to_img_array(array):
    image = Image.fromarray(array, "RGB")
    image.show()
    
my_image = to_array_img("test.jpg")

def padding(X, pad):
    m = X.shape[0]
    nH = X.shape[1]
    nW = X.shape[2]
    nC = X.shape[3]
    def new_length(lng, pad):
        return (lng + (2 * pad))
    
    padded = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)))
    assert(padded.shape == (m,new_length(nH, pad), new_length(nW, pad), nC))
    return padded

padded_image = padding(my_image, 0)
print(padded_image.shape)

def vertical_filter():
    array = np.array([[[1,0,-1], [1,0,-1], [1,0,-1]], 
                      [[1,0,-1], [1,0,-1], [1,0,-1]], 
                      [[1,0,-1], [1,0,-1], [1,0,-1]]])
    assert(array.shape == (3, 3, 3))
    return array

def horizontal_filter():
    array = np.array([[[1,1,1], [0,0,0], [-1,-1,-1]], 
                      [[1,1,1], [0,0,0], [-1,-1,-1]], 
                      [[1,1,1], [0,0,0], [-1,-1,-1]]])
    assert(array.shape == (3, 3, 3))
    return array

def vfilter(X):
    vf = vertical_filter()
    vedges = correlate(X, vf, mode="valid")
    return vedges
def hfilter(X): 
    hf = horizontal_filter()
    hedges = correlate(X, hf, mode="valid")
    return hedges

def convolve(X):
    X = X.reshape(X.shape[1], X.shape[2], X.shape[3])
    filter_1 = vfilter(X)
    filter_2 = hfilter(X)
    output = np.array((filter_1,filter_2))
    return output

edges = convolve(padded_image)

print(edges.shape)

a, b = edges[0], edges[1]

to_img_array(a)

