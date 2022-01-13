FROM supervisely/base-pytorch:6.0.22


# Install PyTorch
RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


# Install other packages


# DETECTRON 2 for CUDA 11.3 and TORCH 1.10

RUN pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html


RUN git clone https://github.com/supervisely-ecosystem/detectron2