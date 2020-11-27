import os
import cv2
import json

"""
This script converts VIA (VGG Image Annotator) JSON files to .txt files required for YOLOv3.
Example:
python convert_json_to_txt.py --images=path_to_images_folder --json=json_file_path --dest=txt files destination folder
--attribute=attribute_name_in_via --labels="{\"label0\":0, \"label1\":1, \"label2\":2}" 
"""


def create_txt_files(via_json_file, images_folder, txt_dest_folder, attribute_name, label_idx_dictionary):
    """
    Input:
    via_json_file = json file exported from VIA annotator
    images_folder = path folder containing images
    txt_dest_folder = destination folder for txt files
    label_idx_dictionary = dictionary whose keys are label names
                        in VIA annotator and values are the
                        corresponding integers to be use for
                        YOLOv3. For example:
                        'label0': 0, 'label1':1, 'label3': 3}
    attribute_name = attribute name used in VIA annotator
    Output:
        Side effects: it creates .txt files whose names correspond
                    to the images names in the train folder.
        None
    """
    data = json.load(open(via_json_file, "r"))
    for (name, file_data) in data.items():
        regions = file_data["regions"]
        image_name = file_data["filename"]
        image_path = os.path.join(images_folder, image_name)
        if len(regions) == 0:
            continue
        try:
            image_height, image_width, _ = cv2.imread(image_path).shape
            for region in regions:
                f = open(os.path.join(txt_dest_folder, os.path.splitext(image_name)[0] + ".txt"), "w+")
                if region["shape_attributes"]["name"] != "rect":
                    print(f"[Warning] Region from image {image_name} contains a non-rectangular region.")
                    f.close()
                    continue
                shape_attributes = region["shape_attributes"]
                x = shape_attributes["x"]
                y = shape_attributes["y"]
                width = shape_attributes["width"]
                height = shape_attributes["height"]
                label_idx = label_idx_dictionary[region["region_attributes"][attribute_name]]
                x_center = x + width / 2
                y_center = y + height / 2
                x_center_norm = x_center / image_width
                y_center_norm = y_center / image_height
                width_norm = width / image_width
                height_norm = height / image_height
                f.write(f"{label_idx} {x_center_norm} {y_center_norm} {width_norm} {height_norm}")
                f.close()
        except AttributeError:
            print(f"[Warning] The image {image_name} does not exist.")
            continue
        except (IOError, SyntaxError):
            print(f"[Warning] Image {image_name} corrupted.")
            continue
    return


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Path to images folder")
    ap.add_argument("--json", required=True, help="Json file path")
    ap.add_argument("--dest", required=True, help="json file destination")
    ap.add_argument("--attribute", required=True, type=str, help="attribute name assigned in VIA annotator")
    ap.add_argument("--labels", required=True, type=json.loads, help="dictionary with string labels as keys "
                                                                     "and values are positive integers")
    args = vars(ap.parse_args())
    print(args["labels"])
    create_txt_files(args["json"], args["images"], args["dest"], args["attribute"], args["labels"])
