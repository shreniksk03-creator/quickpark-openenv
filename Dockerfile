FROM python:3.10

WORKDIR /code

# We use --no-cache-dir to prevent the "Exit Code 1" from happening again
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Explicitly tell the server to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
