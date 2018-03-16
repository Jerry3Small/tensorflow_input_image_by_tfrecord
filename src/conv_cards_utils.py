import numpy as np

# the generated mnist will still get one-hot of 10, needs to be transformed
def transform_to_3(images, labels):
    trans = np.zeros((images.shape[0], 3))
    c = 0
    for e in labels:
        d = 0
        for f in e:
            if d < 2:
                trans[c, d] = labels[c, d]
            elif labels[c, 0] == 0 and labels[c, 1] == 0:
                trans[c, 2] = 1
            else:
                trans[c, 2] = 0
            d += 1
        #if c <= 15:
            #print (e)
            #print (trans[c,:])
        c += 1
    return trans