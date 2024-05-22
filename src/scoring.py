# Functions to combine segmentation masks
from pycocotools import mask as cocomask
import cv2

# Score util
def image_sel(df, images, weights):
    # check if weights cover all categories
    for i, x in enumerate(df['category_id'].unique()):
        if x not in weights.keys():
            weights[x] = 0
    
    # calculate individual scores
    df_score = df.copy()
    cat_nos = df['category_id'].value_counts().to_dict()
    cat_score = {k: sum(cat_nos.values())/v for k, v in cat_nos.items()}
    df_score['cls_score'] = 1 - df_score['score']
    df_score['abun_score'] = 1
    df_score['cat_score'] = df_score.apply(lambda p: cat_score[p['category_id']] * weights[p['category_id']], axis=1)

    # calculate final score on image level
    for i, x in enumerate(df_score['image_id'].unique()):
        df_concern = df_score[df_score['image_id']==x]
        cls_avg_score = 1 * df_concern['cls_score'].mean()
        cls_avg_conf = 1 - cls_avg_score
        nos_pred = len(df_concern)
        abun_score = 0.1 * df_concern['abun_score'].sum()
        cat_pred = df_concern['category_id'].to_list()
        cat_score = 1 * df_concern['cat_score'].mean()
        total_score =  cls_avg_score + abun_score + cat_score
        new_columns = ['class avg. conf', 'nos predictions', 'categories', 'cls_score', 'abun_score', 'cat_score', 'total_score']
        add_list = [cls_avg_conf, int(nos_pred), str(cat_pred), cls_avg_score, abun_score, cat_score, total_score]
        for ind, col in enumerate(new_columns):
            images.loc[images['id'] == x, col] = add_list[ind]
    # df_score.apply(lambda x: len(df_score[df_score['image_id']==x['image_id']]), axis=1)
    # # get image_id with the most number of instances
    # image_id = df['image_id'].value_counts().idxmax()
    # # get the image with the image_id
    # image = images[images['id']==image_id]
    return df_score, images