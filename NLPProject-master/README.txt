Steps to run Project:
1. Unzip ProjectFinalSubmission-TextHarmony.zip.
2. Copy and paste in your Pycharm IDE.
3. Install all the necesary dependencies like NLTK, Wordnet, Spacy.
	Once the spacy has been installed run the following command in terminal
	'python -m spacy download en_core_web_sm'

4. Run SentenceSimilarity.py file which consists of model.
	svr-model.sav will be stored in current folder. (Our Model)
	path = "./data/test-set.txt"
    	model_file_name = 'svr-model.sav'

	By calling:	load_model_on_test(path, model_file_name) will generate sample_predictions.txt file.

5. Use this sample_predictions.txt file for evaluation.py
