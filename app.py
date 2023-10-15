from flask import Flask, request,make_response
from engine import TextImprovmentEngine
import pprint
import json 

app = Flask(__name__)
engine = TextImprovmentEngine("input/Standardised terms.csv")

@app.route('/set_standard_phrases', methods=['POST'])
def set_standard_phrases():
    input_path = request.form['path']
    engine.set_standard_phrases(input_path)
    return make_response(f"standard phrases were updated from {input_path} ", 200)

@app.route('/analyze', methods=['POST'])
def analyze():
    input_txt_path = request.form['input_txt_path']
    similarity_threshold = float(request.form['similarity_threshold'])
    top_per_sentence = int(request.form['top_per_sentence'])
    
    all_results = engine.process_input(input_txt_path, similarity_threshold, top_per_sentence)
    pprint.pprint(all_results)
    return json.dumps(str(all_results))

if __name__ == '__main__':
    app.run(debug=True)