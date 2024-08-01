import os
import subprocess
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from threading import Thread
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def main(old_video_path, coordinates_list):

    prompts = {}  # hold all the clicks we add for visualization

    video_path = f"/content/{old_video_path}"
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_checkpoint = "/content/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    video_dir = "video_dir"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    video_dir = "/content/video_dir"

    command = f"ffmpeg -i {video_path} -q:v 2 -start_number 0 {video_dir}/'%05d.jpg'"
    subprocess.run(command, shell=True, check=True)

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)

    # Process each coordinate
    for coordinates in coordinates_list:
        x, y, current_time = coordinates['x'], coordinates['y'], coordinates['time']
        ann_frame_idx = coordinates['frame']  # Assuming this is intentional
        ann_obj_id = coordinates['ann_obj_id']  # Example object ID

        label_data=coordinates['label']

        points = np.array([[x, y]], dtype=np.float32)
        labels = np.array([label_data], np.int32)

        prompts[ann_obj_id] = points, labels
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

    # Create figure without axes
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    frame = Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))
    ax.imshow(frame)
    show_points(points, labels, ax)
    for i, out_obj_id in enumerate(out_obj_ids):
        show_points(*prompts[out_obj_id], ax)
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), ax, obj_id=out_obj_id)

    # Save the plot without borders
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output_path = f"output_frame_{ann_frame_idx}.png"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    plt.close()
    # Clear the current figure to prepare for the next one
    plt.clf()


    # Propagate the mask throughout the entire video
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Save the propagated frames
    output_frames_dir = "/content/output_frames"
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)

    print("saving the propogated frames")

    for out_frame_idx, frame_name in enumerate(frame_names):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        frame = Image.open(os.path.join(video_dir, frame_name))
        ax.imshow(frame)
        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, ax, obj_id=out_obj_id)
        output_frame_path = os.path.join(output_frames_dir, f"{out_frame_idx:05d}.png")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_frame_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.clf()

    # Merge the frames into a video
    output_video_path = "/content/output_video.mp4"
    command = f"ffmpeg -framerate 30 -i {output_frames_dir}/%05d.png -c:v libx264 -pix_fmt yuv420p {output_video_path}"
    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    coordinates_list = [
        {'x': 61, 'y': 88, 'time': 0.723832,'ann_obj_id':1,'label':1,'frame':0},
        {'x': 560, 'y': 159, 'time': 1.231422,'ann_obj_id':2,'label':1,'frame':0},
        {'x': 61, 'y': 88, 'time': 0.723832,'ann_obj_id':1,'label':0,'frame':150}
        # Add more coordinates as needed
    ]
    old_video_path = "trimmed_0.mp4"
    main(old_video_path, coordinates_list)
