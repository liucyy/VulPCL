import sys
from gensim.models import Word2Vec
import pandas as pd
from deepwalk_embed.Rwalker import randomwalker
sys.path.append('.')

class deepwalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):
        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}
        self.walker = randomwalker(graph)
        self.sentences = self.walker.simutate_walks(num_walks=num_walks, 
                                                    walk_length=walk_length,
                                                    workers=workers,
                                                    verbose=1)
    
    def train(self, embed_size=300, window_size=5, workers=1, iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        model = Word2Vec(**kwargs)
        print("Learning emmbedding vectors done!")
        self.w2v_model = model
        return model
    
    def get_embedding(self,):
        if self.w2v_model == None:
            print("wv2_model not train!")
            return {}
        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]
        
        return self._embeddings
