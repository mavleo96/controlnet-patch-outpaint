#!/bin/bash

# Example usage: ./scripts/download_coco_dataset.sh /data/vmurugan/datasets

ROOT=$1
mkdir -p $ROOT/coco

# Download images
wget http://images.cocodataset.org/zips/train2017.zip -O $ROOT/coco/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip -O $ROOT/coco/val2017.zip
# wget http://images.cocodataset.org/zips/test2017.zip -O $ROOT/coco/test2017.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O $ROOT/coco/annotations_trainval2017.zip

# Unzip images
unzip $ROOT/coco/train2017.zip -d $ROOT/coco/
unzip $ROOT/coco/val2017.zip -d $ROOT/coco/
# unzip $ROOT/coco/test2017.zip

# Unzip annotations
unzip $ROOT/coco/annotations_trainval2017.zip -d $ROOT/coco/