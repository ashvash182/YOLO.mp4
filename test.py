import requests
import time

# Some basic tests for the API

# set up the base URL for the API
BASE_URL = 'http://127.0.0.1:5000'

# define some sample input data
source_name = "dwayne_video"
source_url = "https://storage.googleapis.com/sieve-public-videos/celebrity-videos/dwyane_basketball.mp4"

# test the /push endpoint
def test_push():
    response = requests.post(f"{BASE_URL}/push?" +"source_url=" + source_url)
    assert response.status_code == 200
    assert "id" in response.json()

    # store the job ID for later tests
    job_id = response.json()["id"]
    return job_id

# test the /status endpoint
def test_status(job_id):
    response = requests.get(f"{BASE_URL}/status?id=" + job_id)
    status = response.json()["status"]
    assert response.status_code == 200
    assert status in ["queued", "processing", "finished"]
    return status

# test the /query endpoint
def test_query(job_id):
    response = requests.get(f"{BASE_URL}/query?id=" + job_id)
    assert response.status_code == 200

    # verify that the returned data has the correct format
    data = response.json()
    for key in data:
        assert "class" in data[key]
        assert "positions" in data[key]
        for pos in data[key]["positions"]:
            assert "frame_number" in pos

# test the /list endpoint
def test_list():
    response = requests.get(f"{BASE_URL}/list")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

# run the tests
if __name__ == "__main__":
    job_id = test_push()
    print('pushed job')
    while (True):
        if (test_status(job_id)=="finished"):
            break
        # will arbitrarily check for task completion every 10 seconds
        time.sleep(10)
    test_query(job_id)
    test_list()
    print('testing successful')