import os
import xml.etree.ElementTree as ET

DATASET_TRAIN_PATH = "../dataset/train/"
DATASET_VAL_PATH = "../datasetval/"


def remove_non_target_labels(xml_dir):
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            file_path = os.path.join(xml_dir, xml_file)
            tree = ET.parse(file_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                name = obj.find('name').text
                if name != 'car':
                    root.remove(obj)

            tree.write(file_path)


remove_non_target_labels(DATASET_TRAIN_PATH + 'Annotations')
remove_non_target_labels(DATASET_VAL_PATH + 'Annotations')
