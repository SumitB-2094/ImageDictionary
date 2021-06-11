# ImageDictionary
Image Dictionary using Tensorflow and python.

Image Dictionary is an python based application which enables us to search for a particular image (lets say it as Query Image) from the image dataset, and displays the most accurate images along with accuracy value related to query image, using Machine Learning Techniques and Python libraries. This is based on **"Content based image retrieval"** concepts.

**model_trainer.py** : extracts the feature of image dataset(eg.vehicles) and store it as a numpy file in folder specified (eg.vehicle features).
**Query_loader_and_results** : search for the relevent images related to query image from dataset.
(Query images contains the example query images to be loaded...)

Lets summarize, here are the steps,
● Prepare the dataset (vehicles in this case)
● Extract features from image database (vehicle features)
● Insert the query image and extract its features using query_loader_and_results (eg. 1.car, 2.cycle or 3.ather )
● Calculate the similarities with all images (automatically model will do)
● Retrieve the most similar result (eg. if 1.car then, car displays car and if 2.cycle then, displays cycles)
To specify the architecture, we will use VGG-16 architecture and pretrained weight from the ImageNet.

