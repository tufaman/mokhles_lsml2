# Final Project LSML2 - By Mostafa Mokhles
## Dataset
The dataset used in this project is Amazon Reviews dataset https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews which contains 2 files ```train.csv``` containing 3.6 Million reviews and their sentiment (i.e. Positive or Negative), and ```test.csv``` containing 400k reviews and their sentiment 

## Analysis & Pre-processing
The first step is data analysis and pre-processing, for compute capacity reasons, this was done on Google Colab, and the notebook is attached here in the repository ```Google_Colab.ipynb```. The notebook uses a ```kaggle.json``` API key to download the dataset from Kaggle, and parse the data. Each review in the dataset has a ```Heading``` and ```Text``` coulmns. The notebook uses ```transformers``` library from Huggingface in order to encode each review. The tokenizer and model used for encoding is BERT. The ouput is an embedding for both the heading and the review, which is a numpy vector of size 768. This is done for both the train and the test data and the resulting 4 numpy arrays ```X_train, X_test, y_train, y_test``` are saved in compressed numpy array format ```dataset.npz```
## Model Training
The second step is training a linear regression model on the embeddings obtained from the first step, this is done using MLflow Project run on conda environment. The required ```MLproject``` file and ```conda.yaml``` are present here in the repository, in addition to the training script ```train.py```. They all need to be present in the same folder, along with ```dataset.npz``` file for embeddings. The project uses 2 parameters ```max_iter``` and ```solver```, the metrics being logged are ```accuracy, prescision, recall, f1_sccore``` since this is a binary classification task, in the end the best result was obtained when setting the ```max_iter=500``` and ```solver=liblinear```. The project can be run using the command ```$mlflow run amazon_reviews -P max_iter=500 -P solver=liblinear``` and adding an experiment name can be optional. The obtained results using these parameters are as below:
```
Logistic Regression (Max Iterations=500, solver=liblinear)
  accuracy: 0.8752992021276595
  precision: 0.8788309584421595
  recall: 0.8738617311548703
  f1: 0.8763393004318729
```
These results were obtained from only a smaller version of ```dataset.npz``` which  was around ~300Mb, if using the entire dataset, the embeddings size will be ~7GB and due to lack of memory it couldn't be done. The MLflow project also outputs one file which is the trained model file ```model.pkl``` this is copied to the next step in which we will serve this model.
## Model Serving
The model serving is done using Flask for the web app, Celery for asynch requests, HTML with Jinja templates for the web frontend and Docker for containerizing the app. All the required files are contained within the ```Docker``` folder in the repository. The image was built using ```docker-compose```, the ```docker-compose.yaml``` and ``` Dockerfile``` are ncessary for building the image, the command used to build it is ```docker-compose up```, the netrypoint for the container runs the flask app and docker compose forwards the used port 5000 to the local machine, so after the container is up and running, the page can be accesses through ```localhost:5000/predict``` This opens a web-page where the user can input a new review heading and text, the application will pre-process it and encode it using bert tokenizer and model, then pass it to the classifier model to give predictions (Positive or Negative).
To easily run the app, the image is pushed to dockerhub and available for running using the following command:
```docker run -it -p5000:5000 mostafamokhles/amazon_reviews-web:latest```
