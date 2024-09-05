import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


def load_data(test=False, size=3000, test_size=630, single_img=False, single_img_path='give a path please'):


    if (test):
        size = test_size

    if(single_img):
        size = 1

    for i in range(0, size):

        img_path = 'data/train/images/train_' + str(i) + '.png'
        if (test):
            img_path =  'data/test/images/test_' + str(i) + '.png'

        if (single_img):

            img_path = single_img_path
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            cv2.imwrite('data/single/single_img.png',img)
            img = image.load_img('data/single/single_img.png')
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            y = None
            return x, y

        img = image.load_img(img_path)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        if (i == 0):
            X = x
            continue
        X = np.vstack((X, x))



    for i in range(0, size):


        txt_path = 'data/train/landmarks/train_' + str(i) + '.txt'
        if (test):
            txt_path = 'data/test/landmarks/test_' + str(i) + '.txt'



        with open(txt_path, 'r') as f:
            lines_list = f.readlines()

            for j in range(3, 58):
                string = lines_list[j]
                str1, str2 = string.split(' ')
                x_ = float(str1)
                y_ = float(str2)
                if (j == 3):
                    temp_x = np.array(x_)
                    temp_y = np.array(y_)
                    continue


                temp_x = np.hstack((temp_x, x_))
                temp_y = np.hstack((temp_y, y_))


        if (i == 0):
            Y = np.hstack((temp_x, temp_y))
            Y = Y[None, :]
            continue

        temp = np.hstack((temp_x, temp_y))
        temp = temp[None, :]
        Y = np.vstack((Y, temp))

    return X, Y
