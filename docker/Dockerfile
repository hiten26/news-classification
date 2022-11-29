FROM python:3.7.4
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

#COPY dbpedia_csv/train.csv ./dbpedia_csv/train.csv
#COPY dbpedia_csv/test.csv ./dbpedia_csv/test.csv
#COPY training.py ./training.py

#RUN python3.7 training.py
COPY download_model.py ./download_model.py
COPY models ./models

RUN --mount=type=secret,id=my_secret . /run/secrets/my_secret \
&& python3.7 download_model.py -i ${AWS_ACCESS_KEY} -a ${AWS_SECRET_KEY}


EXPOSE 8501
COPY . /app
ENTRYPOINT [ "streamlit", "run" ]
CMD ["app.py"]