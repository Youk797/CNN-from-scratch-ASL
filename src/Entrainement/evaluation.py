from ..Cnn.numpy_cupy import xp


def accuracy(probs, y):

    pred = xp.argmax(probs, axis=1)  # pour chaque exemple, prend l'index de la proba la plus grande -> classe prédite
    return xp.mean((pred == y).astype(xp.float32))  # compare prédiction et vraie classe, puis fait la moyenne
