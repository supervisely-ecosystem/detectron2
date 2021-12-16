FROM supervisely/base-pytorch:6.0.22


# Install PyTorch
RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html


# Install other packages


# DETECTRON 2 for CUDA 11.1 and TORCH 1.8
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
# RUN pip install detectron2 https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/detectron2-0.6%2Bcu111-cp39-cp39-linux_x86_64.whl

# ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache


RUN git clone https://github.com/supervisely-ecosystem/detectron2
# RUN pip install -r /detectron2/requirements.txt
