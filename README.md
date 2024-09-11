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

# Training

Training with train.py requires the .yaml config file. It lists out the directory for the images and annotation for training, validation and testing. The images are envisaged to be put in
```<this dir>/datasets/<your proj name>/images``` and annotation .txt files (1 .txt per image with the file name of your image) in ```<this dir>/datasets/<your proj name>/labels```.
(Note the yaml file automatically reads ```<this dir>/datasets``` so what you need are the extensions beyond it)

The outputs will be in ```<this dir>/runs/segment/<train name>```. The ```<train name>``` of the output directory can be specified in the name field of ```model.train(...)``` in ```train.py```. The output directory contains various performance curves and result tables against the validation set and the weights are located at ```<this dir>/runs/segment/<train name>/weights```

# Predictions

There are several inputs required in ```predict.py```.

* model - the trained prediction model
* test_dir - the Project Directory, assuming your testing images will be there
* img_path - the Directory containing all the testing images
* output_fn - the filename of the output pseudolabels (in COCO json format)

If you would like to rank images with their informativeness (classification confidence scores, category abundance and instance abundance), you can set ```sel_image = True``` with the desired ```cat_weight``` for each category.

If you would like to visualise the predictions on your testing images, you can set ```vis_pred = True``` to output .jpg to ```output_vis_dir```.
