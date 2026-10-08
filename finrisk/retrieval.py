"""Date-filtered cosine retrieval for precomputed FinBERT document vectors."""
import numpy as np
class DatedRetriever:
    def __init__(self,embeddings,timestamps,documents=None):
        self.emb=np.asarray(embeddings,dtype=np.float32)
        self.times=np.asarray(timestamps)
        self.docs=documents
        self.emb/=np.maximum(np.linalg.norm(self.emb,axis=-1,keepdims=True),1e-12)
    def search(self,query,asof,k=5):
        # Equiv. dense inner-product ranking; optional FAISS may replace this.
        valid=np.flatnonzero(self.times<=asof)
        if len(valid)==0:return []
        q=np.asarray(query,dtype=np.float32);q=q/max(np.linalg.norm(q),1e-12)
        scores=self.emb[valid]@q
        chosen=valid[np.argsort(scores)[-k:][::-1]]
        return [(int(i),float(self.emb[i]@q),None if self.docs is None else self.docs[i]) for i in chosen]
