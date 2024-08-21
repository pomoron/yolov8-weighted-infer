from ultralytics import YOLO
import src.json_util as ju
import src.combine_mask as cm
import src.plot as plot
import src.scoring as sc
import pandas as pd
import os, json

# inputs
model = YOLO('./runs/segment/tcr-nb-230514/weights/best.pt')      # trained pavement model
home_dir = os.path.expanduser("~")
# test_dir = './datasets/a14-o/val/'
test_dir = os.path.join(home_dir, 'PTrans/orthoimage/images/SA-EB-240718')
# img_path = os.path.join(test_dir,'images')
img_path = test_dir
output_fn = os.path.join(test_dir,'SA-EB-240718_infer.json')
# input image selection
sel_image = True
cat_weight = {1: 0, 2: 0, 3: 5, 4: 5, 5:0}
# input visualisation
vis_pred = True
output_vis_dir = os.path.join(test_dir, 'vis_SA-EB-240718_infer')

# Run batched inference on a list of images
results = model(img_path, stream=True, conf=0.05)      # return a list of Results objects

# create dataframes
images = pd.DataFrame(columns=["id", "file_name", "width", "height"])
df_columnsTitles = ["id", "image_id", "category_id", "bbox", "area", "segmentation", "score"]
df = pd.DataFrame(columns=df_columnsTitles)
category = pd.DataFrame(columns=["category_id", "name"])  # Initialize category as an empty DataFrame

# Process results list
# can be done by model.predict(<img/path>, save=True, imgsz=320, conf=0.5) but try this to avoid the yolo txt formats
# iterate to unpack the results object. Objects are divided by image
for i, result in enumerate(results):
    try:
        result_sum = result.summary()  # returns a list of instance data for each image
    except:
        print(f"No bbox predicted or other errors in opening {result.path}")
        continue
    
    # set categories in the first run
    if category.empty:
        category_list = [{"category_id": k+1, "name": v} for k, v in result.names.items()]
        category = pd.DataFrame(category_list, columns=["category_id", "name"])
    
    # extract masks from result
    if result_sum:
        result_mask = result.masks
        mask_list = []
        for x in result_mask.xy:
            seg_dict = {"x": x[:,0], "y": x[:,1]}
            mask_list.append(seg_dict)
    
    # add image to images df
    img_name = os.path.basename(result.path)
    height, width, _ = result.orig_img.shape
    images.loc[len(images)] = {"id": i+1, "file_name": img_name, "width": width, "height": height}
    
    # add annotations to df
    for j, obj in enumerate(result_sum):
        bbox = ju.process_bbox(obj['box'])
        # segment = ju.process_segment(obj['segments'])
        segment = ju.process_segment(mask_list[j])
        area = ju.calculate_area(segment)
        df.loc[len(df)] = {"id": len(df)+1, "image_id": i+1, "category_id": obj['class']+1, "bbox": bbox, "area": area, "segmentation": segment, "score": obj['confidence']}

    # combine masks - YOLO predicts a lot of masks but sometimes they can be combined
    df_check = df[df['image_id']== i+1]
    if len(result_sum) > 1:
        # print(f"nos. predictions: {len(result_sum)}")
        df_check = cm.df_combine_masks(df_check, height, width)
        df = df.drop(df[df['image_id'] == i+1].index)
        df = pd.concat([df,df_check], ignore_index=True)
        df = df.reindex(columns=df_columnsTitles)
        df['id'] = df.index + 1
        # print(f"nos. prediction after merging masks: {len(df_check)}")

    # visualize the results
    if vis_pred:
        os.makedirs(output_vis_dir, exist_ok=True)
        pred_img_fn = os.path.join(output_vis_dir, img_name)
        # new_plot(img_path, df, category, pred_img_fn)
        plot.new_plot(os.path.join(img_path, img_name), df_check, category, pred_img_fn)
        # result.save(filename=pred_img_fn)  # save to disk

# output image selection
if sel_image and len(df)>0:
    _, images_score = sc.image_sel(df, images, cat_weight)
    images_score = images_score.sort_values(by='total_score', ascending=False)
    images_score.to_csv(os.path.join(test_dir,'images_score.csv'), index=False)

# clean dataframes for json dump
category, df = ju.cleanForJson(category, df)
dict_to_json = {
    "categories": category.to_dict('records'),
    "images": images.to_dict('records'),
    "annotations": df.to_dict('records')
    }
with open(output_fn, "w") as outfile:
    json.dump(dict_to_json, outfile, cls=ju.NpEncoder)

del category, images, df