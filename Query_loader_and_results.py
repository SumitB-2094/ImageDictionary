# Import the libraries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Extractor class extracts the feature of each image passed to it as a parameter
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



fe = Extractor()
# storing numpy features into list, can be used later to search for query image features
features = []
# storing path of images , which later can be used to display image as a result
img_paths = []
for feature_path in Path(r"C:\Users\ssbak\Desktop\vehicles feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path(r"C:\Users\ssbak\Desktop\vehicles") / (feature_path.stem + ".jpg"))
features = np.array(features)


# Insert the image query
img = Image.open(r"C:\Users\ssbak\Desktop\3.ather.jpg")
# Extracting its features
query = fe.extract(img)
dists = np.linalg.norm(features - query, axis=1)
ids = np.argsort(dists)[:5]
scores = [(dists[id], img_paths[id]) for id in ids]

# MATPLOTLIB
#visulize the results
axes=[]
fig=plt.figure(figsize=(9,9))
axes.append(fig.add_subplot(5,6,1))
axes[0].set_title(str('QUERY'))
plt.imshow(img)
# plt.xlim(200,600)
# plt.ylim(350,150)

for a in range(5):
    score = scores[a]
    axes.append(fig.add_subplot(5, 6, a+2))
    subplot_title=str(score[0])
    axes[-1].set_title(subplot_title)
    plt.axis('off')
    plt.imshow(Image.open(score[1]))
fig.tight_layout()
plt.show()
plt.close()