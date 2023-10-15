from utils import *
from semantic_search import SemanticSearch
from paraphrasing import Paraphrasing

class TextImprovmentEngine:
    def __init__(self, standard_phrases_filename) -> None:
        self.control_phrases = read_csv(standard_phrases_filename)
        self.semantic_search_model = SemanticSearch()
        self.paraphrasing_model = Paraphrasing() 

    def set_standard_phrases(self,filename):
        self.control_phrases = read_csv(filename) 

        
    def process_input(self,input_text_path,threshold,max_results):
        text = read_text(input_text_path)
        sentences =  convert_text_to_list_sentences(text)
        all_results = []
        for sentence in sentences: 
            sub_sens = find_ngrams(sentence)
            cos_res = self.semantic_search_model.calculate_similarity(sub_sens, self.control_phrases)
            top_k = self.semantic_search_model.get_top_k(cos_res,max_results)
            for top_index in top_k : 
                score = cos_res[top_index[0],top_index[1]]
                if score > threshold: 
                    rewritten_sentence = self.paraphrasing_model.paraphrase(sentence,sub_sens[top_index[0]],self.control_phrases[top_index[1]])[0]
                    all_results.append({"orignal_sentence" : sentence, 
                                        "control_phrase":self.control_phrases[top_index[1]],
                                        "similarity_score":score,
                                        "subtext_matched":sub_sens[top_index[0]],
                                        "paraphrased_result" : rewritten_sentence
                                        })
        return all_results


if __name__ =="__main__":  
    eng = TextImprovmentEngine("input/Standardised terms.csv")
    all_results = eng.process_input("input/sample_text.txt", 0.4, 3)
    for res in all_results:
        print(res)