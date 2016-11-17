import numpy as np

def rectify(h):
    h = h.reshape((4,2))
    h_new = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    h_new[0] = h[np.argmin(add)]
    h_new[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    h_new[1] = h[np.argmin(diff)]
    h_new[3] = h[np.argmax(diff)]

    return h_new
