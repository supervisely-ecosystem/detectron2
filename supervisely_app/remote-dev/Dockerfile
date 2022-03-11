ARG IMAGE
FROM $IMAGE


# Install PyTorch
RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


# Install other packages


# DETECTRON 2 for CUDA 11.3 and TORCH 1.10

RUN pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
# RUN pip install detectron2 https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/detectron2-0.6%2Bcu111-cp39-cp39-linux_x86_64.whl

# ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache


RUN git clone https://github.com/supervisely-ecosystem/detectron2
# RUN pip install -r /detectron2/requirements.txt


RUN apt-get update && apt-get install -y openssh-server
EXPOSE 22

RUN apt-get install -y sudo
RUN mkdir -p /run/sshd



# MORE SECURE: use a trusted RSA key
ARG home=/root
RUN mkdir $home/.ssh
COPY authorized_keys $home/.ssh/authorized_keys
RUN chown root:root $home/.ssh/authorized_keys && \
    chmod 600 $home/.ssh/authorized_keys

COPY sshd_deamon.sh /sshd_deamon.sh
RUN chmod 755 /sshd_deamon.sh
CMD ["/sshd_deamon.sh"]
ENTRYPOINT ["sh", "-c", "/sshd_deamon.sh"]