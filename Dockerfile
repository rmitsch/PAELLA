# Instructions copied from - https://hub.docker.com/_/python/
FROM python:3.5-alpine

# Copy file with python requirements into container.
COPY setup/requirements.txt /tmp/requirements.txt
# Copy file containing commands for installing numpy versus openblas.
COPY setup/install_numpy_with_openblas.sh /tmp/install_numpy_with_openblas.sh
# Copy setup file.
COPY setup/setup.sh /tmp/setup.sh

# Set versions of numpy and openblas to be installed.
ENV NUMPY_VERSION="1.13.1" \ 
	OPENBLAS_VERSION="0.2.18" 

# Source: https://hub.docker.com/r/o76923/alpine-numpy-stack/~/dockerfile/
RUN echo "http://alpine.gliderlabs.com/alpine/v3.5/main" > /etc/apk/repositories && \ 
	echo "http://alpine.gliderlabs.com/alpine/v3.5/community" >> /etc/apk/repositories && \ 
	echo "@edge http://alpine.gliderlabs.com/alpine/edge/community" >> /etc/apk/repositories && \ 
	apk update && \
	apk --no-cache add openblas-dev && \
	apk add ca-certificates && \
	update-ca-certificates && \
	apk add openssl=1.0.2k-r0

RUN export NPROC=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || 1) && \ 
	apk --no-cache add --virtual build-deps \
        musl-dev \
        linux-headers \
        g++ && \ 
	cd /tmp && \ 
	ln -s /usr/include/locale.h /usr/include/xlocale.h && \ 
	pip install cython && \ 
	cd /tmp && \ 
	#wget https://sourceforge.net/projects/numpy/files/NumPy/$NUMPY_VERSION/numpy-$NUMPY_VERSION.tar.gz && \ 
	wget https://github.com/numpy/numpy/releases/download/v$NUMPY_VERSION/numpy-$NUMPY_VERSION.tar.gz && \
	tar -xzf numpy-$NUMPY_VERSION.tar.gz && \ 
	rm numpy-$NUMPY_VERSION.tar.gz && \ 
	cd numpy-$NUMPY_VERSION/ && \ 
	cp site.cfg.example site.cfg && \ 
	echo -en "\n[openblas]\nlibraries = openblas\nlibrary_dirs = /usr/lib\ninclude_dirs = /usr/include\n" >> site.cfg && \ 
	python -q setup.py build -j ${NPROC} --fcompiler=gfortran install && \ 
	cd /tmp && \ 
	rm -r numpy-$NUMPY_VERSION
	#pip install numexpr pandas scipy && \ 
	

# Install custom TOPAC dependencies.
RUN apk update && \
	# Install various drivers necessary on alpine for some of the python dependencies.
	apk add postgresql-dev=9.6.4-r0 && \
	apk add zlib-dev=1.2.11-r0

# Install python dependencies.
RUN	pip install -r /tmp/requirements.txt

# Allow execution of setup script.
RUN	chmod +x /tmp/setup.sh
	# Run manual setup.
RUN ./tmp/setup.sh && \
	# Remove build dependencies.
	# To test: Does using numpy (gensim, ...) require does files?
	apk --no-cache del --purge build-deps

# Declare which port(s) should be exposed.
EXPOSE 5000

# run the command
CMD ["python", "./source/app.py"]