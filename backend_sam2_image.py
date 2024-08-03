import torch
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, image_path, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.show()

def process_image(coordinates_list,old_video_path):

    video_path=f'/content/downloads/{old_video_path}'

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    video_dir = "/content/video_dir"

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    

    command = f"ffmpeg -i {video_path} -q:v 2 -start_number 0 {video_dir}/'%05d.jpg'"
    subprocess.run(command, shell=True, check=True)

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    coordinates=coordinates_list[-1]

    x, y, current_time = coordinates['x'], coordinates['y'], coordinates['time']

    image_name = f"{round(current_time * 30):05d}" # the frame index we interact with
    label_data=coordinates['label']

    image = Image.open(os.path.join(video_dir,f'{image_name}.jpg'))
    image = np.array(image.convert("RGB"))

    sam2_checkpoint = "/content/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)

    input_point = np.array([[x, y]])
    input_label = np.array([label_data])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    sorted_ind = np.argsort(scores)
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    show_masks(image, masks, scores, image_path=f'/content/processed_image/{image_name}.jpg', point_coords=input_point, input_labels=input_label, borders=True)

    return f'{image_name}.jpg'

def main(coordinates_list,video_path):
    process_image(coordinates_list,video_path)

if __name__ == "__main__":

    coordinates_list = [
        {'x': 61, 'y': 88, 'time': 0.723832,'ann_obj_id':1,'label':1},
        {'x': 560, 'y': 159, 'time': 1.231422,'ann_obj_id':2,'label':1},
        {'x': 61, 'y': 88, 'time': 3.723832,'ann_obj_id':1,'label':0}
        # Add more coordinates as needed
    ]
    old_video_path = "/content/trimmed_0.mp4"
    main(coordinates_list,old_video_path)