#!/bin/bash
# EMR bootstrap action — installs the Python dependencies on every cluster node
# (master + core/task). Referenced when launching the EMR cluster so the
# featurization UDF can run TensorFlow/Pillow on each worker.
#
# Upload this to S3 and point the EMR "bootstrap action" at it, e.g.:
#   aws s3 cp scripts/bootstrap-emr.sh s3://<BUCKET>/bootstrap-emr.sh
#   then: EMR -> Create cluster -> Bootstrap actions -> s3://<BUCKET>/bootstrap-emr.sh
set -euo pipefail

sudo python3 -m pip install -U setuptools
sudo python3 -m pip install -U pip
sudo python3 -m pip install wheel
sudo python3 -m pip install pillow
sudo python3 -m pip install pandas==1.2.5
sudo python3 -m pip install pyarrow
sudo python3 -m pip install boto3
sudo python3 -m pip install s3fs
sudo python3 -m pip install fsspec
sudo python3 -m pip install tensorflow
