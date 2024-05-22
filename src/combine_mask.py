# Functions to combine segmentation masks
from pycocotools import mask as cocomask
import cv2
import numpy as np

def polygon_to_mask(polygons, height, width):
    '''
    Args:
        polygon (list): in [[...]]
        height, width (int): height and width of image
    Returns:
        np.array: a mask of shape (H, W)
    '''
    the_mask = np.zeros((height, width), dtype=np.uint8)

    # tackle the problem of having >1 polygon in an instance
    if len(polygons) > 1:
        for poly in polygons:
            # poly = np.array(poly).reshape((-1, 2))
            rles = cocomask.frPyObjects([poly], height, width)
            a_mask = np.squeeze(cocomask.decode(rles))
            # print(a_mask.shape)
            the_mask = np.logical_or(the_mask, a_mask)
    else:
        rles = cocomask.frPyObjects(polygons, height, width)
        the_mask = np.squeeze(cocomask.decode(rles))

    return the_mask

def mask_iou(masks):
    """
    Calculate the IoU of a list of masks.
    Args:
        masks (list): a list of masks of shape (N, H, W).
    Returns:
        np.array: outputs of a matrix of shape (len(masks), len(masks)).
    """
    num_masks = len(masks)
    mask_ious = np.zeros((num_masks, num_masks), dtype=np.float32)

    # Calculate the intersection and union of masks using logical AND and OR operation
    for i in range(num_masks):
        for j in range(i+1, num_masks):
            intersection = np.logical_and(masks[i], masks[j])
            union = np.logical_or(masks[i], masks[j])
            iou = np.sum(intersection) / np.sum(union)
            mask_ious[i, j] = iou
            mask_ious[j, i] = iou

    return mask_ious

def mask_to_polygon(maskedArr):
    """
    Convert a mask in a binary matrix of shape (H, W) to a polygon.
    Args:
        maskedArr (np.array): a binary matrix of shape (H,W) that contains True/False value
    Returns:
        segmentation[0] (np.array): an array of the polygon. [x1, y1, x2, y2...]
        area (float): the size of the polygon
        bbox (list): Bounding box coordinates in COCO format [x_min, y_min, delta_x, delta_y].
    """
    # Convert the binary mask to a binary image (0 and 255 values)
    binary_image = maskedArr.astype(np.uint8) * 255

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve the polygon coordinates for each contour
    polygons = []
    for contour in contours:
        if contour.size >= 6:
            polygons.append(contour.flatten().tolist())
        # polygon = contour.squeeze().tolist()
        # polygons.append(polygon)
    # Calculate areas and bbox
    area = np.sum(maskedArr)    
    # RLEs = mask_util.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
    # RLE = mask_util.merge(RLEs)
    # area = mask_util.area(RLE)
    bbox = find_remasked_bbox(maskedArr)

    return polygons, area, bbox

def merge_masks(mask1, mask2):
    """
    Merge two masks.
    Args:
        mask1 (np.array): First mask of shape (H, W).
        mask2 (np.array): Second mask of shape (H, W).
    Returns:
        np.array: Merged mask if IoU is above the threshold, otherwise None.
    """
    merged_mask = np.logical_or(mask1, mask2)
    return merged_mask

# also need to fix areas
def find_remasked_bbox(mask):
    """
    Find the bounding box that fits the resized mask.

    Args:
        mask (np.array): Resized mask of shape (H', W').
        original_shape (tuple): Shape of the original image or mask (H, W).

    Returns:
        list: Bounding box coordinates in COCO format [x_min, y_min, delta_x, delta_y].
    """
    # Find the minimum and maximum coordinates that enclose the mask region
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [x_min, y_min, x_max-x_min, y_max-y_min]

def df_combine_masks(df_check, height_i, width_i, iou_merge_thres=0.25):
    df_check = df_check.reset_index(drop=True)
    seg_polygons = df_check['segmentation'].to_list()
    seg_catid = df_check['category_id'].to_list()

    # # IoU now calculated per iteration because it should be re-calculated after merging each polygon
    # # convert polygons of an image to masks
    # seg_masks = []
    # for idplygn, x in enumerate(seg_polygons):
    #     mask = polygon_to_mask(x, height_i, width_i)
    #     seg_masks.append(mask)
    
    # # calculate iou of all instances

    # masks_iou = mask_iou(seg_masks)
    
    # merge masks
    # for i in range(masks_iou.shape[0]):
        # for j in range(i + 1, masks_iou.shape[1]):
    for i in range(len(seg_polygons)):
        for j in range(i + 1, len(seg_polygons)):
            # create the mask after merging
            concerned_polygons = [df_check.loc[i, 'segmentation'], df_check.loc[j, 'segmentation']]
            if all(concerned_polygons):
                seg_masks = []
                for idplygn, x in enumerate(concerned_polygons):
                    mask = polygon_to_mask(x, height_i, width_i)
                    seg_masks.append(mask)
                the_mask_iou = mask_iou(seg_masks)[0, 1]
                # the_mask_iou = mask_iou(seg_masks)[i,j]
                # dataframe fixing
                if the_mask_iou > iou_merge_thres and seg_catid[i]==seg_catid[j]:
                    # merged_mask = merge_masks(seg_masks[i],seg_masks[j])
                    merged_mask = merge_masks(seg_masks[0],seg_masks[1])
                    merged_polygon = mask_to_polygon(merged_mask)
                    # seg_concat = []
                    # seg_concat[:] = merged_polygon[0]
                    # add the new merged polygon to the i-th entry
                    df_check.at[i, 'segmentation'] = merged_polygon[0]           # new segmentation for the new mask
                    df_check.at[i,'area'] = merged_polygon[1]                     # new area for the new mask
                    df_check.at[i,'bbox'] = find_remasked_bbox(merged_mask)       # new bbox for the new mask
                    # clear j-th entry 
                    df_check.at[j,'segmentation'] = []
                    df_check.at[j,'area'] = np.NaN
                    df_check.at[j,'bbox'] = []
            else:
                continue
    
    df_check = df_check.dropna(subset=['area'])
    return df_check