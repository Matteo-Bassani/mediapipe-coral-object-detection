import os
import json
import tensorflow as tf
import subprocess
import wandb
from mediapipe_model_maker import quantization, object_detector
from libs import object_detector_extended
from keras.callbacks import EarlyStopping

# Settings
DATASET_TRAIN_PATH = "datasettrain/"
DATASET_VAL_PATH = "datasetval/"
EXPORT_PATH = "exported_models/"
TFLITE_FLOAT32_NAME = "model.tflite"
TFLITE_INT8_NAME = "model_int8.tflite"
METADATA_PATH = "metadata.json"
METADATA_H_NAME = "metadata.hpp"
METADATA_H_PATH = os.path.join(EXPORT_PATH, METADATA_H_NAME)

# Hyperparameters
ARCHITECTURE = "MOBILENET_V2_I320"
LR = 0.3
BATCH_SIZE = 32
EPOCHS = 10

# Wandb settings
wandb.init(
    project="Coral Board Transfer Learning",
    config={
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    }
)


# Set Wandb Callback
class WandbCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs)


early_stopping_callback = EarlyStopping(monitor='box_loss', patience=3)

# Extract train and validation data
train_data = object_detector_extended.Dataset.from_pascal_voc_folder(DATASET_TRAIN_PATH)
validation_data = object_detector_extended.Dataset.from_pascal_voc_folder(DATASET_VAL_PATH)

# Load pre-trained model and specify hyperparameters
spec = object_detector_extended.SupportedModels.MOBILENET_V2_I320
hparams = object_detector_extended.HParams(
    learning_rate=LR,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    export_dir=EXPORT_PATH,
)
options = object_detector_extended.ObjectDetectorOptions(
    supported_model=spec,
    hparams=hparams,
)

# Retrain model
model = object_detector_extended.ObjectDetector.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options,
    callbacks=[WandbCallback(), early_stopping_callback]
)

# Evaluate model performance
loss, coco_metrics = model.evaluate(
    validation_data,
    batch_size=4,
)
print(f"Validation loss: {loss}")
print(f"Validation metrics: {coco_metrics}")

# Export 32-bit float model
model.export_model()

# Perform post-training quantization (8-bit integer) and save quantized model
quantization_config = quantization.QuantizationConfig.for_int8(
    representative_data=validation_data,
)
model.restore_float_ckpt()
model.export_model(
    model_name=TFLITE_INT8_NAME,
    quantization_config=quantization_config,
)

# Run a edgetpu compiler
result = subprocess.run(['edgetpu_compiler', '-l', '-o', EXPORT_PATH, os.path.join(EXPORT_PATH, TFLITE_INT8_NAME)],
                        capture_output=True, text=True)

# Import model metadata
with open(METADATA_PATH, 'r') as file:
    metadata = json.load(file)

# Parse metadata
custom_metadata = metadata['subgraph_metadata'][0]['custom_metadata'][0]
anchors = custom_metadata['data']['ssd_anchors_options']['fixed_anchors_schema']['anchors']
num_values_per_keypoint = custom_metadata['data']['tensors_decoding_options']['num_values_per_keypoint']
apply_exponential_on_box_size = custom_metadata['data']['tensors_decoding_options']['apply_exponential_on_box_size']
x_scale = custom_metadata['data']['tensors_decoding_options']['x_scale']
y_scale = custom_metadata['data']['tensors_decoding_options']['y_scale']
w_scale = custom_metadata['data']['tensors_decoding_options']['w_scale']
h_scale = custom_metadata['data']['tensors_decoding_options']['h_scale']

# Figure out when the resets (sectors) occur, the x/y increases, and width/height of anchors
reset_idxs = []
y_strides = []
x_strides = []
widths_per_section = []
widths = []
heights_per_section = []
heights = []
reset_flag = True
x_stride_flag = True
width_flag = True

# Go through all the anchors
num_anchors = len(anchors)
for i in range(num_anchors):

    # Store the first index
    if i == 0:
        reset_idxs.append(i)

    # Only measure strides on not 0 indexes
    else:

        # New section: reset flags
        if anchors[i]['y_center'] < anchors[i - 1]['y_center']:
            reset_idxs.append(i)
            reset_flag = True
            x_stride_flag = True
            width_flag = True

        # Measure Y increase (stride)
        if reset_flag:
            if anchors[i]['y_center'] > anchors[i - 1]['y_center']:
                y_inc = anchors[i]['y_center'] - anchors[i - 1]['y_center']
                y_strides.append(round(y_inc, 5))
                reset_flag = False

        # Measure X increase (stride)
        if x_stride_flag:
            if anchors[i]['x_center'] > anchors[i - 1]['x_center']:
                x_inc = anchors[i]['x_center'] - anchors[i - 1]['x_center']
                x_strides.append(round(x_inc, 5))
                x_stride_flag = False

    # Record widths and heights of the anchor boxes
    if width_flag:
        if i != 0 and anchors[i]['x_center'] > anchors[i - 1]['x_center']:
            widths.append(widths_per_section)
            widths_per_section = []
            heights.append(heights_per_section)
            heights_per_section = []
            width_flag = False
        else:
            width = anchors[i]['width']
            widths_per_section.append(round(width, 5))
            height = anchors[i]['height']
            heights_per_section.append(round(height, 5))

# Calculate the number of sectors
num_sectors = len(reset_idxs)

# Calculate the number of anchors per coordinate
num_anchors_per_coord = len(widths[0])

# Calculate the number of Xs in each Y
num_xs_per_y = []
for sector in range(num_sectors):
    num_xs_per_y.append(int(1.0 / x_strides[sector] * num_anchors_per_coord))

print(f"Number of anchors {num_anchors}")
print(f"Number of sectors: {num_sectors}")
print(f"Number of anchors per coordinate: {num_anchors_per_coord}")
print(f"Reset indexes: {reset_idxs}")
print(f"Number of Xs per Y: {num_xs_per_y}")
print(f"X strides: {x_strides}")
print(f"Y strides: {y_strides}")
print("Widths:")
for wps in widths:
    print(wps)
print("Heights:")
for hps in heights:
    print(hps)

    # Generate header file for metadata information
h_str = f"""\
// Filename: {METADATA_H_NAME}

#ifndef METADATA_HPP
#define METADATA_HPP

namespace metadata {{
    constexpr unsigned int num_anchors = {num_anchors};
    constexpr int apply_exp_scaling = {1 if apply_exponential_on_box_size else 0};
    constexpr float x_scale = {x_scale};
    constexpr float y_scale = {y_scale};
    constexpr float w_scale = {w_scale};
    constexpr float h_scale = {h_scale};
    constexpr unsigned int num_sectors = {num_sectors};
    constexpr unsigned int num_anchors_per_coord = {num_anchors_per_coord};
"""

# Print reset indexes
h_str += "    constexpr unsigned int reset_idxs[] = {\r\n"
h_str += "        "
for i in range(num_sectors):
    h_str += f"{reset_idxs[i]}"
    if i < num_sectors - 1:
        h_str += ", "
h_str += "\r\n"
h_str += "    };\r\n"

# Print the number of X values for each Y value
h_str += "    constexpr unsigned int num_xs_per_y[] = {\r\n"
h_str += "        "
for i in range(num_sectors):
    h_str += f"{num_xs_per_y[i]}"
    if i < num_sectors - 1:
        h_str += ", "
h_str += "\r\n"
h_str += "    };\r\n"

# Print the X strides
h_str += "    constexpr float x_strides[] = {\r\n"
h_str += "        "
for i in range(num_sectors):
    h_str += f"{x_strides[i]}"
    if i < num_sectors - 1:
        h_str += ", "
h_str += "\r\n"
h_str += "    };\r\n"

# Print the Y strides
h_str += "    constexpr float y_strides[] = {\r\n"
h_str += "        "
for i in range(num_sectors):
    h_str += f"{y_strides[i]}"
    if i < num_sectors - 1:
        h_str += ", "
h_str += "\r\n"
h_str += "    };\r\n"

# Print the anchor widths for each section
h_str += f"    constexpr float widths[{num_sectors}][{len(widths[0])}] = {{\r\n"
for i in range(num_sectors):
    h_str += "        {"
    for j in range(len(widths[0])):
        h_str += f"{widths[i][j]}"
        if j < len(widths[0]) - 1:
            h_str += ", "
    h_str += "}"
    if i < num_sectors - 1:
        h_str += ","
    h_str += "\r\n"
h_str += "    };\r\n"

# Print the anchor heights for each section
h_str += f"    constexpr float heights[{num_sectors}][{len(heights[0])}] = {{\r\n"
for i in range(num_sectors):
    h_str += "        {"
    for j in range(len(heights[0])):
        h_str += f"{heights[i][j]}"
        if j < len(heights[0]) - 1:
            h_str += ", "
    h_str += "}"
    if i < num_sectors - 1:
        h_str += ","
    h_str += "\r\n"
h_str += "    };\r\n"

# Close header file
h_str += """\
}

#endif // METADATA_HPP
"""

# write to .h file
with open(METADATA_H_PATH, 'w') as file:
    file.write(h_str)
