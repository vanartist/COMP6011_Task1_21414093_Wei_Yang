import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict


source_json = "D:/coco/annotations/instances_train2017.json"
image_dir = Path("D:/coco/train2017")
output_dir = Path("yolov8_seg_dataset_clean")
max_images = 3000
train_ratio = 0.8


coco_to_seg = {
    1: 0, 2: 1, 3: 2, 4: 3,
    6: 4, 8: 5, 10: 6, 13: 7,
    17: 8, 18: 9, 19: 10, 20: 11, 21: 12
}
target_cat_ids = set(coco_to_seg.keys())


def convert_polygon(segmentation, width, height):
    if not segmentation or not isinstance(segmentation, list):
        return None
    poly = segmentation[0]
    if len(poly) < 6:
        return None
    norm_poly = []
    for i in range(0, len(poly), 2):
        x = poly[i] / width
        y = poly[i + 1] / height
        norm_poly.extend([f"{x:.6f}", f"{y:.6f}"])
    return norm_poly

def main():
    with open(source_json, 'r') as f:
        coco = json.load(f)

    images_info = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)

    for ann in coco["annotations"]:
        if isinstance(ann["segmentation"], list):
            anns_by_image[ann["image_id"]].append(ann)

    image_ids_with_targets = []
    for img_id, anns in anns_by_image.items():
        for ann in anns:
            if ann["category_id"] in target_cat_ids:
                image_ids_with_targets.append(img_id)
                break

    image_ids_with_targets = list(set(image_ids_with_targets))
    random.shuffle(image_ids_with_targets)
    selected_ids = image_ids_with_targets[:max_images]
    selected_images = [images_info[iid] for iid in selected_ids]

    # Train/val split
    num_train = int(train_ratio * len(selected_images))
    train_set = selected_images[:num_train]
    val_set = selected_images[num_train:]
    img_id_to_split = {img["id"]: "train" for img in train_set}
    img_id_to_split.update({img["id"]: "val" for img in val_set})

    # Prepare output dirs
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_kept = 0
    skipped_images = []

    for img in selected_images:
        img_id = img["id"]
        file_name = img["file_name"]
        width, height = img["width"], img["height"]
        split = img_id_to_split[img_id]
        anns = anns_by_image.get(img_id, [])

        lines = []

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in target_cat_ids:
                continue
            poly = convert_polygon(ann["segmentation"], width, height)
            if not poly:
                continue
            class_id = coco_to_seg[cat_id]
            lines.append(f"{class_id} " + " ".join(poly))


        if not lines:
            skipped_images.append(file_name)
            continue

        src_img = image_dir / file_name
        dst_img = output_dir / "images" / split / file_name
        dst_lbl = output_dir / "labels" / split / file_name.replace(".jpg", ".txt")

        if src_img.exists():
            shutil.copy(src_img, dst_img)
            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines))
            total_kept += 1
        else:
            print(f"Image not found: {src_img}")


    if skipped_images:
        with open(output_dir / "skipped_images.txt", "w") as f:
            for name in skipped_images:
                f.write(name + "\n")

if __name__ == "__main__":
    main()

