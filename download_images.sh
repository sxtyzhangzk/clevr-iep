#!/bin/sh
set -e
wget https://github.com/sxtyzhangzk/clevr-dataset-gen/releases/download/v1/images.zip
unzip images.zip -d benchmark/
