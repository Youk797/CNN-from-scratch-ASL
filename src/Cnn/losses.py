from .numpy_cupy import xp
from .utils import softmax

class SoftmaxM:
    def __init__(self):
        # On garde en mémoire ce qui sera utile pour le backward
        self.probs = None  # probabilités calculées par softmax (N, K)
        self.y = None      # vrais labels (N,)

    def forward(self, logits, y):
        # logits : sortie brute du modèle (N, K)
        # y      : labels 
        self.y = y 

        #  Convertit les logits en probabilités 
        probs = softmax(logits, axis=1)  
        self.probs = probs                # on garde pour le backward

        # Calcule la loss 
        # on prend la proba de la bonne classe pour chaque exemple
        N = logits.shape[0]               # nombre d'exemples dans le batch
        p = probs[xp.arange(N), y]        # p[i] = proba de la bonne classe pour l'exemple i

        # loss = moyenne de -log(p)
        # + 1e-12 pour éviter log(0)
        loss = -xp.mean(xp.log(p + 1e-12))
        return loss

    def backward(self):
        # Renvoie le gradient de la loss par rapport aux logits 
        N, K = self.probs.shape          

        d = self.probs.copy()             # on part de probs
        d[xp.arange(N), self.y] -= 1.0    # on soustrait 1 sur la colonne de la vraie classe
        d /= N                            # moyenne sur le batch

        return d.astype(xp.float32)       # gradient final (N, K)
