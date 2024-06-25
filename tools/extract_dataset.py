from settings import *
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import os
import shutil
import random


# Function to copy images and their annotations
def move_files(image_list, target_dir, images_dir, annotations_dir):
    for image in image_list:
        image_path = os.path.join(images_dir, image)
        annotation_path = os.path.join(annotations_dir, image.replace('.jpg', '.xml'))

        # Copy image
        shutil.move(image_path, os.path.join(target_dir, 'images', image))

        # Copy annotation
        shutil.move(annotation_path, os.path.join(target_dir, 'Annotations', image.replace('.jpg', '.xml')))


def extract_dataset(name, origin, object_class):
    dataset_name = "dataset-" + name
    export_path = DATASETS_PATH + dataset_name

    dataset = foz.load_zoo_dataset(
        origin,
        split="train",
        label_types=["detections"],
        classes=[object_class],
        shuffle=True,
        label_field="ground_truth",
        dataset_name=dataset_name,
        max_samples=MAX_SAMPLES,
        seed=SEED,
    )

    bbox_area = F("bounding_box")[2] * F("bounding_box")[3]

    dataset_view = dataset.filter_labels("ground_truth", F("label").is_in(object_class))
    dataset_view = dataset_view.filter_labels("ground_truth", bbox_area >= MIN_AREA)
    dataset_view = dataset_view.filter_labels("ground_truth", bbox_area <= MAX_AREA)

    session = fo.launch_app(dataset_view)

    dataset_type = fo.types.VOCDetectionDataset

    dataset_view.export(
        export_dir=export_path,
        dataset_type=dataset_type,
        label_field="ground_truth",
    )

    # Create directories for images and annotations
    annotations_dir = os.path.join(export_path, 'labels')
    images_dir = os.path.join(export_path, 'data')

    # Create directories for train and validation splits
    train_dir = os.path.join(export_path, 'train')
    validation_dir = os.path.join(export_path, 'validation')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Get list of all images
    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    # Shuffle the list of images
    random.shuffle(images)

    # Split images into train and validation sets (80/20)
    split_index = int(TRAIN_SPLIT * len(images))
    train_images = images[:split_index]
    validation_images = images[split_index:]

    # Ensure the subdirectories for JPEGImages and Annotations exist
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'Annotations'), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, 'Annotations'), exist_ok=True)

    # Copy the files
    move_files(train_images, train_dir, images_dir, annotations_dir)
    move_files(validation_images, validation_dir, images_dir, annotations_dir)

    os.rmdir(annotations_dir)
    os.rmdir(images_dir)

    print(f'Total images: {len(images)}')
    print(f'Train images: {len(train_images)}')
    print(f'Validation images: {len(validation_images)}')

    session.wait()
