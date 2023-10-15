from transformers import BartForConditionalGeneration, BartTokenizer
import torch

class Paraphrasing:
    def __init__(self, model_name="eugenesiow/bart-paraphrase") :
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.margin_token = 3

    def generate(self,text): 
        """generate paraphrasing

        Args:
            text (string): _description_
            max_length (int): maximum number of tokens in the output sentence

        Returns:
            string: rephrashed sentence
        """
        input_ids = self.tokenizer(
                text,
                return_tensors="pt", 
                truncation=True,
            ).input_ids
            
        outputs = self.model.generate(
                input_ids.to(self.device),
                do_sample = True,
                top_k = 50,
                top_p=0.95,
                temperature=0.9,
                max_length= len(text.split( )) + self.margin_token
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True) 
    
    def paraphrase(self,text,to_replace,control): 
        """replace part of the text with control sentence and parphrase the output

        Args:
            text (string): text to operate on
            to_replace (string): part of the text which we want to replace with control sentence
            control (string): control sentece to use

        Returns:
            string: paraphrased text after replacing part of it with control text 
        """
        text = text.replace(to_replace, control)
        res = self.generate(text)
        return res 