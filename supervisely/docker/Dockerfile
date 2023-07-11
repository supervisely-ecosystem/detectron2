FROM supervisely/base-pytorch:6.0.22


# Install PyTorch
# RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install other packages
COPY dev_requirements.txt dev_requirements.txt
RUN pip3 install -r dev_requirements.txt

# DETECTRON 2 for CUDA 11.3 and TORCH 1.10
RUN pip3 install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html


RUN git clone https://github.com/supervisely-ecosystem/detectron2
