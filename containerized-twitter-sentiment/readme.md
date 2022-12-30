# Containerized Twitter Sentiment Analysis

###  Get the environment ready
* Inside the project folder create virtual environment using venv and activate it
    * python -m venv env
    * source env/bin/activate

* creating requirements.txt 
    ```
        numpy
        pandas
        scikit-learn
        flask
        nltk
        regex
    ```
* python -m pip install -r requirements.txt

### Build the ML Model
* download sentiment.tsv file
* copy the training script in train.py
```python
import pandas as pd 
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

def remove_pattern(input_txt,pattern):
    '''
    removes pattern from input_txt using regex
    '''
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    
    ## removes punctuations
    input_txt = re.sub(r'[^\w\s]', ' ', input_txt)

    return input_txt.strip().lower()


if __name__ == '__main__':
	## loading data
	data = pd.read_csv("sentiment.tsv",sep = '\t')
	data.columns = ["label","body_text"]

	# Features and Labels
	data['label'] = data['label'].map({'pos': 0, 'neg': 1})

	data['clean_tweet'] = np.vectorize(remove_pattern)(data['body_text'],"@[\w]*")
	tokenized_tweet = data['clean_tweet'].apply(lambda x: x.split())

	stemmer = PorterStemmer()
	tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 

	for i in range(len(tokenized_tweet)):
	    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

	data['clean_tweet'] = tokenized_tweet

	X = data['clean_tweet']
	y = data['label']

	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data

	#from sklearn.model_selection import train_test_split
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	## Using Classifier
	clf = LogisticRegression()
	clf.fit(X,y)
	
	## save vectorizer and model
	with open('logistic_clf.pkl', 'wb') as f:
    	    pickle.dump((cv,clf), f)
```
* run train.py
    * python train.py
    * it will train and save the model as a pickel file

### Front end
* create the home template and save as home.html
```html 
<!DOCTYPE html>
<html>
<head>
	<title>Home</title>
</head>
<body>

	<header>
		<div class="container">
			<div id="brandname">
				<h2 align="center">Machine Learning Sentiment Analysis Application Containerization using Docker</h2>
			</div>
			<hr/>
			<br/>
		<h2 align="center">Twitter Sentiment Analysis</h2>
		</div>
	</header>

	<div class="ml-container" align="center">

		<form action="{{ url_for('predict')}}" method="POST">
		<p>Enter Your Message Here</p>
		<!-- <input type="text" name="comment"/> -->
		<textarea name="message" rows="4" cols="60"></textarea>
		<br/>
		<br/>
		<input type="submit" class="btn-info" value="predict">
		
	</form>
		
	</div>

</body>
</html>
```

* Create result.html with the following code
```html
<!DOCTYPE html>
<html>
<head>
	<title>Sentiment Analysis</title>
</head>
<body>

	<header>
		<div class="container">
			<div id="brandname">
				<h3 align="center">Sentiment Analysis Result</h3>
			</div>
		<hr/>
		<br/>
		</div>
	</header>

	<p style="color:blue;font-size:20;text-align: center;"><b>Sentiment</b></p>
	<div class="results" align="center">
		
		{% if prediction == 1%}
		<h2 style="color:red;">Negative</h2>
		{% elif prediction == 0%}
		<h2 style="color:blue;">Positive</h2>
		{% endif %}
		
		<a href="..">
		<input type="submit" value="Go Back..">
		</a>
	</div>

</body>
</html>
```

### Create flask app
* Use the following code to create the flask app
```python
from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)

def remove_pattern(input_txt,pattern):
    '''
    removes pattern from input_txt using regex
    '''
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)

    ## removes punctuations
    input_txt = re.sub(r'[^\w\s]', ' ', input_txt)

    return input_txt.strip().lower()


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        clean_test = remove_pattern(message,"@[\w]*")
        tokenized_clean_test = clean_test.split()
        stem_tokenized_clean_test = [stemmer.stem(i) for i in tokenized_clean_test]
        message = ' '.join(stem_tokenized_clean_test)
        data = [message]
        data = cv.transform(data)
        my_prediction = clf.predict(data)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	
	##initialize stemmer	
	stemmer = PorterStemmer()

	##load vectorizer and model
	with open('model/logistic_clf.pkl', 'rb') as f:
	    cv, clf = pickle.load(f)
	
	app.run(host='0.0.0.0',port=5000)
```
* Run the following command to start the app
    * python app.py

### Containerization
* start docker desktop
* in the terminal open app folder and run the following command
    * docker build -t twitter-sentiment:v1 .
* run the app
    * docker run -p5001:5001 twitter-sentiment:v1

## Upload to docker hub

* login to docker from cli
    * docker login
    * docker tag twitter-sentiment:v1 shoaib6174/twitter-sentiment:v1.0.0
    * docker push  shoaib6174/twitter-sentiment:v1.0.0

* pulling the images
    * docker pull shoaib6174/twitter-sentiment:v1.0.0

```
docker tag local-image:tagname new-repo:tagname
docker push new-repo:tagname
```