FROM arvslat/amazon-sagemaker-cookbook-python-base:1

COPY train /usr/local/bin/train
COPY serve /usr/local/bin/serve
COPY config.json /var/tmp/config.json

RUN apt update
RUN apt install -y gcc python3-dev libgomp1

RUN pip3 install --upgrade pip wheel pandas>=0.25.3 flask
RUN pip3 install lightfm>=1.16
RUN apt remove -y binutils binutils-common binutils-x86-64-linux-gnu cpp cpp-7 dh-python gcc gcc-7 gcc-7-base libasan4 libatomic1 libbinutils libc-dev-bin libc6-dev libcc1-0 libcilkrts5 libexpat1-dev libgcc-7-dev libisl19 libitm1 liblsan0 libmpc3 libmpfr6 libmpx2 libpython3-dev libpython3.6 libpython3.6-dev libquadmath0 libtsan0 libubsan0 linux-libc-dev manpages manpages-dev python3-dev python3.6-dev
