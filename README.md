INSTALLATIONS
1. Run pip install -r requirements.txt

2. Should also have YOLO and SORT installed (git clone https://github.com/ultralytics/yolov5 ; git clone https://github.com/abewley/sort), along with all associated requirements (pip install -r requirements.txt in both above directories)

RUNNING THE API
1. To install and intialize redis server, run the following commands in Ubuntu (via WSL if on Windows)

    $ sudo apt-get install redis
    $ sudo service redis-server start

    Optional, to test server is up and running:
    $ redis-cli
    $ ping (Should return PONG on 127.0.0.1:6379)

4. 'flask run' in powershell to start Flask app (http://127.0.0.1:5000)

5. In separate powershell, 'celery -A main.celery worker --pool=solo --loglevel=INFO' to start celery queue

API should be running with celery task queuing and redis datastore+message brokering. POST request status+response can be viewed in the Celery terminal with execution time and output. Flask terminal will display API messages with appropriate response codes.

TESTING
test.py will check for:
    1. appropriate reponse codes
    2. job_id being passed in /push requests
    3. appropriate status codes for pushed jobs
    4. structure of dictionary output
    5. listing of pushed jobs
    6. will output "testing successful" upon completion

Dockerization in progress...
