import cv2
import numpy as np
from featureExtraction import LBP
from skimage.feature import local_binary_pattern
from model import DNN_Model,SVM_Model
print(1%10)

def cutImage(image,m,n):
    h, y = image.shape
    # z = 0
    code_h, code_w = h // m, y // n
    codeimage = np.zeros((m * n, code_h * code_w))

    for i in range(0, m):
        for j in range(0, n):
            codeimage[i*m+j] = np.reshape((image[i*code_h: (i+1)*code_h, j*code_w: (j+1)*code_w]),(1,-1))
    return np.int64(codeimage),code_h,code_w


def hist(codeimage, m, n, code_h, code_w):
    hist_ = np.zeros((m * n, 256))
    for i in range(m * n):
        for j in range(code_h * code_w):
            hist_[i][codeimage[i][j]] += 1
    return hist_/(code_h*code_w)

def hist2(codeimage, m, n, code_h, code_w):
    hist_

if __name__ == '__main__':
    m = 10
    n = 10
    feature = []
    for i in range(1,166,1):
        img = cv2.imread("yaleFaceDatabase/s" + str(i) + ".bmp", 0)
        lbp = local_binary_pattern(img,8,2)
        codeimage,code_h,code_w = cutImage(lbp, m, n)
        hist_ = hist(codeimage,m,n,code_h,code_w)
        feature.append(np.reshape(hist_,(1,-1))[0])
    feature = np.int64(feature)

    lable = ['center light', 'glasses', 'happy', 'left light', 'no glasses', 'normal', 'right light', 'sad', 'sleepy', 'surprised', 'wink']
    y_id = np.zeros(165)
    y = np.zeros((165,11))
    for i in range(165):
        y_id[i] = i % 11
        y[i][i % 11] = 1

    print(feature.shape, y.shape)
    shuffle_index = np.random.permutation(165)

    x, y = feature[shuffle_index], y[shuffle_index]



    x_train, y_train = x[:132], y[:132]
    x_test, y_test = x[132:], y[132:]

    y_id = y_id[shuffle_index]
    y_train_id = y_id[:132]
    y_test_id = y_id[132:]

    #DNN
    # model = DNN_Model(25600, 500, 11,loadModel=True,modeFile="model0416.h5")
    # model.train(x_train,y_train,epochs=50000)
    # print(model.evaluate(x_test, y_test))
    # model.saveModel("model0416.h5")

    #SVM
    model = SVM_Model()
    model.train(x_train,y_train_id)
    print(model.evaluate(model.predict(x_test), y_test_id))









