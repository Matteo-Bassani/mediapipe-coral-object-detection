import tools as fo
import tools.zoo as foz
from tools import ViewField as F

CLASS = "car"

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes = [CLASS],
    shuffle=True,
    label_field="ground_truth",
    dataset_name="open-images-car",
    #max_samples=10000,
    #seed=51,
)

bbox_area = F("bounding_box")[2] * F("bounding_box")[3]

test_view = dataset.filter_labels("ground_truth", F("label").is_in(CLASS))
test_view = test_view.filter_labels("ground_truth", bbox_area >= 0.001)
test_view = test_view.filter_labels("ground_truth", bbox_area <= 0.5)

session = fo.launch_app(test_view)
session.wait()

dataset_type = fo.types.VOCDetectionDataset

dataset.export(
    export_dir="../dataset/train",
    dataset_type=dataset_type,
    label_field="ground_truth", 
)
