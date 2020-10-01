# Cough Detection Website

The website will be the front end and provide APIs so that any application can interact with the model

## Stack Used
1) Flask: Python-based web framework
2) Heroku: Deployment platform

## API Paramters

### 1. Inference Endpoint

The endpoint will support classifying either sound or image coughing data. However, the paramter has to be correct in order for the classification to be made on the correct type (image or sound).

Live Endpoint: `https://cough-classification.herokuapp.com/inference` (Unstable)
Localhost Endpoint: `http://localhost:5000/inference`

| Parameter     | Required      | Description    | Example    |
| ------------- |:-------------:| :------------: | :--------  |
| url           | True          | Public URL of the sound or image file    |   https://cough-images.s3-ap-southeast-2.amazonaws.com/S003_v007_frame_27.jpg    |
| type          | True          | To specify whether its a sound or image     | `sound` or `image`    |

Example: http://localhost:5000/inference?type=image&url=https://cough-images.s3-ap-southeast-2.amazonaws.com/S003_v007_frame_27.jpg

### 2. Data Endpoint

This endpoint will be an interface to receive information of the nodes that are used in the system. It will get the stored information from the DynamoDB and returns it in a JSON response. The main aim for this endpoint is so that websites or applications can make use of the data to visualize the data. Other relevent purpose for this data can also be used such as machine learning to predict the trends once the database has been more populated.

Live Endpoint: `https://cough-classification.herokuapp.com/data/` (Unstable)
Localhost Endpoint: `http://localhost:5000/data/`

| Parameter     | Required      | Description    | Example    |
| ------------- |:-------------:| :------------: | :--------  |
| id      | True          | The node id that has a history of cough detection and other metadata of that node    | 2     |

Example: http://localhost:5000/data/1

## Steps to Run the Server on Localhost (venv)

1. Install `Python 3`, `pip` and `venv`
2. Clone the repository
3. Go inside the repository `cd cough-detection`
4. Create the virtual environment `python3 -m venv .`
5. Activate Virtual Environment `source bin/activate`
6. Install all the dependencies `pip install -r requirements.txt`
7. Run application `python app.py`
8. Go to `http://localhost:5000/`