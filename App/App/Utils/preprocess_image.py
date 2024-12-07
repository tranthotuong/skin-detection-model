from tensorflow.keras.preprocessing import image as IG
import numpy as np
from PIL import Image

# Function to preprocess the image
def resize_image(image):
    # Resize the image to 244x244 and 299x299
    img_244 = image.resize((224, 224))
    img_244 = IG.img_to_array(img_244)
    img_244 = np.expand_dims(img_244, axis=0)

    img_299 = image.resize((299, 299))
    img_299 = IG.img_to_array(img_299)
    img_299 = np.expand_dims(img_299, axis=0)

    return img_244, img_299