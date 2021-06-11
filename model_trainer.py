
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path
import numpy as np
from PIL import Image

class Extractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    def extract(self, img):
        #vgg16 Arch. neccesarry inputs
        img = img.resize((224, 224))
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)

#extract features from our dataset containing images
fe = Extractor()
# Path of image dataset. and extension of images
for img_path in sorted(Path(r"C:\Users\ssbak\Desktop\vehicles").glob("*.jpg")):
    print(img_path)
    # Extract Features and saving as numpy files of each image in our dataset
    feature = fe.extract(img=Image.open(img_path))
    feature_path =  Path(r"C:\Users\ssbak\Desktop\vehicles feature") / (img_path.stem + ".npy")
    np.save(feature_path, feature)#will save image feature as a numpy file in feature path