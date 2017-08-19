# Source: https://github.com/prakhar1989/docker-curriculum

# Instructions copied from - https://hub.docker.com/_/python/
FROM jfloff/alpine-python:3.4

# Copy file with python requirements into container.
COPY requirements.txt /tmp/requirements.txt
# Copy setup file.
COPY setup.sh /tmp/setup.sh

# Install dependencies.
RUN apk update && \
	# Install postgresql-dev necessary for psycopg2.
	apk add postgresql-dev=9.6.3-r0 && \
	pip install -r /tmp/requirements.txt \
	# Manual setup steps.
	./tmp/setup.sh


# Declare which port(s) should be exposed.
EXPOSE 5000

# run the command
CMD ["python", "./source/app.py"]