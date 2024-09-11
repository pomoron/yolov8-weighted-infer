# Weighted Inference via YOLOv8

Train and make inference with Ultralytics YOLOv8 by ```pip install ultralytics``` in your anaconda environment.

Details of the package and the YOLOv8 powered by Ultralytics can be found in <url>https://github.com/ultralytics/ultralytics</url>

Apart from simple training and inference codes for instance segmentation tasks, the predict.py in this repo aims to provide the following functions:
* calculate the values of an image by counting instances and weight against the predicted category (set ```sel_image==True``` and ```cat_weight``` dict)
* combine overlapping masks into a unified mask (via ```src.combine_mask.py``` incorporated in ```predict.py```)
* output the annotated images (set ```vis_pred == True```)

These will be useful in selecting a small batch of interesting images from a large dataset for annotation.

# Main Packages to install

The operations are assumed to be performed in an Anaconda environment with Python. Codes in this repo require several packages to be installed (via pip).
```
* ultralytics      # the main package for YOLOv8 train and predict
* pycocotools      # for merging masks and converting between masks and polygons
```
