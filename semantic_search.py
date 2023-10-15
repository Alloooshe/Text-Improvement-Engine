
import numpy as np 
import torch 
from sentence_transformers import SentenceTransformer, util
from utils import k_largest_index

class SemanticSearch:
    def __init__(self, model_name='msmarco-distilbert-base-v4'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name,device=device)
        
    def calculate_similarity(self,queries,docs):
        """calculate semantic similarity between queries and docs.

        Args:
            queries (list[string]): list of queries to use.
            docs (list[string]): list of docs to use.

        Returns:
            numpy.array: 2d numpy array that contains cosine similarity score for all docs and all queries.
        """
        queries_encoding = self.model.encode(queries)
        docs_encoding = self.model.encode(docs)
        cos_sim = util.cos_sim(queries_encoding,docs_encoding).numpy()
        return cos_sim    
    
    def get_top_k(self,cos_sim,k):
        """get top k pairs (query,doc) with highest similarity score

        Args:
            cos_sim (numpy.array): 2d numpy array that contains cosine similarity score for all docs and all queries.
            k (uint): top ranking 

        Returns:
             numpy.array: indecies of top k element.
        """
        return k_largest_index(cos_sim,k)

    def find_top(self, cos_sim):
        """find pair (query,doc) with highest similarity score

        Args:
            cos_sim (numpy.array): 2d numpy array that contains cosine similarity score for all docs and all queries.

        Returns:
            numpy.array: [[x,y]] indecies of max element.
        """
        top_sims =  k_largest_index(cos_sim,1)
        return top_sims
    
    