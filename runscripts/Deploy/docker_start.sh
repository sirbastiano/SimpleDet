export TAG=openmmlab/mmdeploy:ubuntu20.04-cuda11.3-mmdeploy
# docker pull $TAG # run this if you want to update the image
# docker run -v $(pwd):/root/workspace/mmdet/ -it --rm $TAG
docker run -v $(pwd):/root/workspace/mmdet/ --gpus=all -it --rm $TAG