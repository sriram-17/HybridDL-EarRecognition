import matplotlib.pyplot as plt

def put_landmarks(i, pred, single_img=False):

    img_path = 'data/test/images/test_' + str(i) + '.png'
    img_result_path = 'data/test/results/result_' + str(i) + '.png'

    if(single_img):
        img_path = 'data/single/sampleimage.png'
        img_result_path = 'data/single/result/result.png'

    img_original = plt.imread(img_path)

    for j in range(0,55):
        plt.scatter([pred[j]], [pred[j+55]])


    plt.imshow(img_original)
    plt.savefig(img_result_path)
    plt.close()
