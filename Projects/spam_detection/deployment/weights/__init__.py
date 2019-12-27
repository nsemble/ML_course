import os
import pickle
MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) #absolute path for model

with open(os.path.join(MODEL_DIR,'spam_detection_mnb.sav'),'rb') as model_file:
	model = pickle.loads(model_file.read())

with open(os.path.join(MODEL_DIR,'tfidf_mnb.sav'),'rb') as tfidf_file:
	tfidf = pickle.loads(tfidf_file.read())
		
with open(os.path.join(MODEL_DIR,'vocabulary_mnb.sav'),'rb') as vocab_file:
	vocab = pickle.loads(vocab_file.read())

label_mapping = {'ham': 0, 'spam': 1}
	
