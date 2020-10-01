# Cough Detection Website

The website will be the front end and provide APIs so that any application can interact with the model

## Stack Used
1) Flask: Python-based web framework
2) Heroku: Deployment platform

## API Paramters

1. Inference Endpoint

The endpoint will support classifying either sound or image coughing data. However, the paramter has to be correct in order for the classification to be made on the correct type (image or sound).

Endpoint: `https://cough-classification.herokuapp.com/inference/`

| Parameter     | Required      | Description    | Example    |
| ------------- |:-------------:| :------------: | :--------  |
| url           | True          | Public URL of the sound or image file    |   https://s3.cough-detection.ap-2.com/test-image.jpg    |
| type          | True          | To specify whether its a sound or image     | `sound` or `image`    |


2. Data Endpoint

This endpoint will be an interface to receive information of the nodes that are used in the system. It will get the stored information from the DynamoDB and returns it in a JSON response. The main aim for this endpoint is so that websites or applications can make use of the data to visualize the data. Other relevent purpose for this data can also be used such as machine learning to predict the trends once the database has been more populated.

Endpoint: `https://cough-classification.herokuapp.com/data/`

| Parameter     | Required      | Description    | Example    |
| ------------- |:-------------:| :------------: | :--------  |
| id      | True          | The node id that has a history of cough detection and other metadata of that node    | 2     |