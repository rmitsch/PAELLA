FROM rmitsch/alpine-python-gensim

##########################################
# 1. Copy relevant files into container.
##########################################

# Copy file with python requirements into container.
COPY setup/requirements.txt /tmp/requirements.txt
# Copy setup file.
COPY setup/setup.sh /tmp/setup.sh
# Copy source code.
COPY source /source

# Allow execution of setup scripts.
RUN chmod +x /tmp/setup.sh

##########################################
# 2. Install dependencies.
##########################################

RUN apk update && \
	# cmake for installing dependencies.
	apk add cmake=3.6.3-r0 && \
	# git for pulling Multicore-t-SNE from git.
	apk add git=2.11.3-r0 && \
	# Install various drivers necessary on alpine for some of the python dependencies.
	apk add libffi-dev=3.2.1-r3 && \
	apk add postgresql-dev=9.6.4-r0 && \
	apk add zlib-dev=1.2.11-r0 && \
	apk add libxml2=2.9.4-r3 && \
	apk add libxml2-dev=2.9.4-r3 && \
	apk add libxslt-dev=1.1.29-r1 && \
	# Install python dependencies.
	pip install -r /tmp/requirements.txt && \
	# Execute additional setup and clean up build environment.
	./tmp/setup.sh && \
	# Remove build dependencies.
	# To test: Does using numpy (gensim, ...) require build files?
	apk --no-cache del --purge build-deps && \
	apk del cmake && \
	apk del git

##########################################
# 3. Launch server.
##########################################

# Declare which port(s) should be exposed.
EXPOSE 5000

# run the command
CMD ["python", "./source/app.py"]