#pulls dockerfile (with env and OS) from https://hub.docker.com/_/python
FROM python:3

# set working directory
WORKDIR /usr/src/app

# copy file from your machine into new machine's current work directory
COPY requirements.txt .

# install libraries defined in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy files from your machine into new machine
COPY . .

# run command in new machine
CMD ["python", "app.py"]
