#!/usr/bin/env bash

LOCATION=$1
BATCH_SIZE=$2

echo "Downloading data"
gsutil -m cp -R gs://gqn-dataset/rooms_ring_camera $LOCATION

echo "Deleting small records"
TRAIN_PATH="$LOCATION/rooms_ring_camera/train"
find "$TRAIN_PATH/*.tfrecord" -type f -size -10M | xargs rm # remove smaller than 10mb

echo "Converting data"
python tfrecord-converter.py $LOCATION rooms_ring_camera -b $BATCH_SIZE -m "train"
echo "Training data: done"
python tfrecord-converter.py $LOCATION rooms_ring_camera -b $BATCH_SIZE -m "test"
echo "Testing data: done"