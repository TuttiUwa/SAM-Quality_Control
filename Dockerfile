FROM python:3.9
RUN pip install --upgrade pip
RUN pip install opencv-python-headless==4.5.3.56
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
WORKDIR /SAM-Quality_Control
COPY . /SAM-Quality_Control



