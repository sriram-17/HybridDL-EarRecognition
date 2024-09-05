from my_CNN_model import load_current_model
from utilities import load_data
from landmarks import put_landmarks

model = load_current_model('my_model')

single_img=False
X, Y = load_data(test=True, test_size=630, single_img=single_img, single_img_path='data/single/single_img.png')


for i in range(0,len(X)):

    temp = X[i]
    temp = temp[None,:]
    prediction = model.predict(temp)
    print(temp,"\nnew\n",prediction)
    for p in range(len(prediction[0])):

        prediction[0][p] = int(prediction[0][p] * 224)

    put_landmarks(i, prediction[0], single_img=False)