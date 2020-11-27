# VIA to YOLOv3
This script converts VIA ([VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/)) JSON files to .txt files required for [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3#annotation-folder).

## Usage 
```
python via_to_yolo.py --images=path_to_images_folder --json=json_file_path --dest=txt files destination folder --attribute=attribute_name_in_via --labels="{\"label0\":0, \"label1\":1, \"label2\":2}"
```

This will automatically create txt files in the destination folder needed to train a YOLOv3 model. 
