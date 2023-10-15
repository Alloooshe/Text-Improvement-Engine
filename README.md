# Text-Improvement-Engine
a tool that analyses a given text and suggests improvements based on the similarity to a list of "standardised" phrases.

# How it works
1- we split the input text into sentences using a pretrained model then we split again using regex.
2- we gnerate all posible word ngrams with n>=2 for each sentence, this is to imporve the semantic search.
3- we apply sentence embedding model and cosine similarity to find the best matches between any ngram and any standard "control" sentence.
4- we replace the ngram with the standard phrase and we use a paraphrasing model to regenerate the sentence

# Tech and models
1- sentence split using spacy (you should download en_core_web_lg model)
2- sentence similarity BERT.
3- paraphrasing BERT
4- everything works behind Flask app, it is an overkill, you can use through curl or postman, anyway I print the results in consol.

# How to use 
* install everything with pip install -r requirements.txt
* install en_core_web_lg using !python -m spacy download en_core_web_lg
* run flask app
* first run will be slow as we download BERT models 
* you can change the default standard phrases by calling POST http://127.0.0.1:5000/set_standard_phrases with path in dataforms
* call the analysis functionality using  POST http://127.0.0.1:5000/analyze which takss three arguments : 
    - input_txt_path (/path/to/input.txt)
    - similarity_threshold (float)
    - top_per_sentence (int)