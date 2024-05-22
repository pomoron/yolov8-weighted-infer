# plot with new bbox and mask
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np

def new_plot(image_path, df, category, file_name):
    # Assuming image_path is the path to your image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Get image dimensions
    height, width, _ = image.shape
    # Create a new figure with a single subplot and set figure size to match image dimensions
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(width / dpi, height / dpi), dpi=dpi)
    cmap = mpl.colormaps['tab10']

    # Display the image
    ax.imshow(image)
    
    # Assuming df is your DataFrame and it has columns 'bbox' for bounding boxes, 'segmentation' for masks, 'category_id' for category labels, and 'score' for confidence scores
    for i, row in df.iterrows():
        plot_color = cmap(row['category_id'])      # same colour for the same category

        # Draw bounding box
        bbox = row['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=plot_color, facecolor='none')    # params: (x,y), width, height
        ax.add_patch(rect)

        # Draw mask
        mask = row['segmentation'][0]
        mask = np.array(mask)               # Convert the list to a numpy array
        poly = patches.Polygon(mask.reshape(-1, 2), fill=True, color=plot_color, alpha=0.4)
        ax.add_patch(poly)

        # Add label and score
        category_name = category[category['category_id'] == row['category_id']]['name'].values[0]
        label = f"{category_name}: {row['score']:.2f}"
        plt.text(bbox[0], bbox[1], label, color='white', fontsize=4, bbox=dict(facecolor=plot_color, edgecolor=plot_color, boxstyle='square,pad=0'))

    ax.axis('off')    
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=dpi)  # Save the plot as an image
    plt.close(fig)