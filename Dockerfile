FROM continuumio/miniconda3
RUN apt update -y
RUN apt upgrade -y
RUN mkdir /usr/src/mgr
WORKDIR /usr/src/mgr
COPY data ./data
COPY lib ./lib
COPY models ./models
COPY routes ./routes
COPY static ./static
COPY templates ./templates
COPY app.py .
COPY spec-file.txt .
RUN conda install --file spec-file.txt -y
ENTRYPOINT [ "python", "-u", "app.py" ]
