FROM python:3.5-alpine

##########################################
# 1. Copy relevant files into container.
##########################################

# Copy file with python requirements into container.
COPY setup/requirements.txt /tmp/requirements.txt
# Copy file containing commands for installing numpy versus openblas.
COPY setup/install_numpy_with_openblas.sh /tmp/install_numpy_with_openblas.sh
# Copy setup file.
COPY setup/setup.sh /tmp/setup.sh
# Copy source code.
COPY source /source

##########################################
# 2. Prepare repos and env. variables.
##########################################

# Set versions of numpy and openblas to be installed.
ENV NUMPY_VERSION="1.13.1" \ 
	OPENBLAS_VERSION="0.2.18" 

# Allow execution of setup scripts.
RUN chmod +x /tmp/install_numpy_with_openblas.sh && \
	chmod +x /tmp/setup.sh

##########################################
# 3. Install numpy with openblas.
##########################################

# Run setup scripts.
RUN ./tmp/install_numpy_with_openblas.sh

##########################################
# 4. Install custom TOPAC dependencies.
##########################################

RUN apk update && \
	# Install various drivers necessary on alpine for some of the python dependencies.
	apk add postgresql-dev=9.6.4-r0 && \
	apk add zlib-dev=1.2.11-r0 && \
	apk add libxml2=2.9.4-r3 && \
	apk add libxml2-dev=2.9.4-r3 && \
	apk add libxslt-dev=1.1.29-r1
	# Install python dependencies.


#RUN	pip install -r /tmp/requirements.txt

	# Execute additional setup and clean up build environment.
#RUN	./tmp/setup.sh && \
	# Remove build dependencies.
	# To test: Does using numpy (gensim, ...) require build files?
#	apk --no-cache del --purge build-deps

##########################################
# 5. Launch server.
##########################################

# Declare which port(s) should be exposed.
EXPOSE 5000

# run the command
CMD ["python", "./source/app.py"]