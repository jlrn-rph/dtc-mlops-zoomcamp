FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["starter.py", "model.bin", "./"]

RUN mkdir -p "output"
CMD ["python", "starter.py", "2021", "4", "model.bin"]