from PIL import ImageFont
import visualkeras
from keras.models import load_model

model = load_model('my_model.h5')

font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
visualkeras.layered_view(model, legend=True, font=font)  # font is optional!

