from matplotlib import image
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from PIL import Image
import pandas as pd

if __name__ == "__main__":

    """
    anstatt (1, image)
    (n_samples, image)
    exclude one for testingf
    """
    inputs = []
    targets = []
    step = 0
    height = 128
    width = 128
    for img_name in listdir("stem_data_cropped_container/stem_data_cropped"):
        step+=1
        if step%5 == 0:
            break
        input_r = Image.open("stem_data_cropped_container/stem_data_cropped/" + img_name)
        target_r = Image.open("stem_lbl_cropped_container/stem_lbl_cropped/" + img_name)
        #input = np.random.rand(10,10,3)
        input = input_r.resize((width, height))
        input = np.asarray(input)
        target = target_r.resize((width, height))
        target = np.asarray(target)
        input = input.flatten()
        #target = np.random.randint(0,2,(10,10,3))
        target = target.flatten()
        input = input.reshape(1, -1)
        target = target.reshape(1, -1)
        inputs.append(input)
        targets.append(target)
    X = np.vstack([i for i in inputs])
    y = np.vstack([t for t in targets])
    print(X.shape, "x shape")
    print(y.shape, "y shape")
    clf = RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=8)
    clf.fit(X, y)
    print("fittet")
    test_r = Image.open("stem_data_cropped_container/stem_data_cropped/img7_crop_0.png")
    #test = np.rando
    test = test_r.resize((width, height))
    test = np.asarray(test)
    test = test.flatten()
    test = test.reshape(1,-1)
    print(test.shape, "shape of test")
    ytest = clf.predict(test)
    result = np.reshape(ytest, (128,128,3))
    result = result.astype(np.int32)
    print(result)
    print(result.shape)
    plt.imshow(result)
    plt.show()
    #result = np.reshape(ytest, (128,128,3))
    #print(ytest)
    #visualize_tree(clf, input, target, boundaries=False)
