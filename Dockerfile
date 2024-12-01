# Usa un'immagine Python ufficiale
FROM python:3.10.15

#update of some libraries because it gave me problems
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copia i file del progetto nella directory di lavoro
COPY ./requirements.txt /MedRobotLab/requirements.txt

WORKDIR /MedRobotLab

# Installa le dipendenze del progetto
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt