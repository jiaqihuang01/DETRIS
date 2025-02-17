import os
import shutil
from tqdm import tqdm
root = "./datasets/masks/"
datasets = ['refcoco', 'refcoco+', 'refcocog_u']

for dataset in datasets:
    mixed_dataset_dir = os.path.join(root, "refcoco_mixed")
    os.makedirs(mixed_dataset_dir, exist_ok=True)

for dataset in datasets:
    source_dir = os.path.join(root, dataset)
    img_lists = os.listdir(source_dir)
    for img in tqdm(img_lists, desc=f"{dataset}"):
        source_file = os.path.join(source_dir, img)
        target_file = os.path.join(mixed_dataset_dir, f"{dataset}_{img}")
        shutil.copyfile(source_file, target_file)
