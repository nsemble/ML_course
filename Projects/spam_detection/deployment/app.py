from flask import Flask, render_template,url_for,request
import pickle
from weights import model, tfidf, vocab


app = Flask(__name__)


@app.route('/')

def home():
	return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
	# label_mapping = {'ham': 0, 'spam': 1}

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		# intent = [k for k,v in label_mapping.items() for i in model.predict(vocab.transform([message]).toarray())  if v == i ]
		intent = model.predict(vocab.transform(data).toarray())
	return render_template('result.html',prediction = intent[0])


if __name__ == '__main__':
	app.run(debug=True,host='0.0.0.0',port=5000)

