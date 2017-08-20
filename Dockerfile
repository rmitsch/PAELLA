# Source: https://github.com/prakhar1989/docker-curriculum

# Instructions copied from - https://hub.docker.com/_/python/
FROM jfloff/alpine-python:3.4

# Copy file with python requirements into container.
COPY requirements.txt /tmp/requirements.txt
# Copy setup file.
COPY setup.sh /tmp/setup.sh

# Install dependencies.
RUN apk update && \
	# Install various drivers necessary on alpine for some of the python dependencies.
	apk add postgresql-dev=9.6.4-r0 && \
	apk add zlib-dev=1.2.11-r0 && \
	# Install python dependencies.
	pip install -r /tmp/requirements.txt && \
	# Allow execution of setup script.
	chmod +x /tmp/setup.sh && \
	# Run manual setup.
	RUN ./tmp/setup.sh



# Declare which port(s) should be exposed.
EXPOSE 5000

# run the command
CMD ["python", "./source/app.py"]