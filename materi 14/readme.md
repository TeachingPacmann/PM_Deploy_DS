How to run:

- Create conda env for python 3.8
- ``pip install -r requirements.txt``
- Run this ``gunicorn app.main:app --worker-class uvicorn.workers.UvicornWorker --reload``
- Open localhost:8000/docs
- Try out the the endpoint predict_house/v1