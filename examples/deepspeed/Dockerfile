FROM determinedai/environments:cuda-11.1-pytorch-1.9-lightning-1.3-tf-2.4-gpu-a173dcd
RUN apt-get install -y pdsh

RUN pip uninstall -y horovod

ARG HOROVOD_PIP=horovod
ARG HOROVOD_WITH_TENSORFLOW=1
ARG HOROVOD_WITH_PYTORCH=1
ARG HOROVOD_WITHOUT_MXNET=1
ARG HOROVOD_GPU_OPERATIONS=NCCL
ARG HOROVOD_WITHOUT_MPI=1
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    pip install "$HOROVOD_PIP" && \
        ldconfig 
RUN pip install ninja psutil packaging attrdict
RUN locate nccl| grep "libnccl.so" | tail -n1 | sed -r 's/^.*\.so\.//'

# Install MMDetection
ARG ALWAYS_RUN
RUN git clone https://github.com/determined-ai/deepspeed
WORKDIR /deepspeed
RUN git checkout determined
RUN python setup.py install
