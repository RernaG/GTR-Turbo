import numpy as np    
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_self_bleu(sentences, n_gram=2):  
    if len(sentences) < 2:
        return 1      
    bleu_scores = []
    smoothing = SmoothingFunction().method1

    for i in range(len(sentences)):
        candidate = sentences[i].split()
        references = [s.split() for j, s in enumerate(sentences) if j != i]
        score = sentence_bleu(references, candidate, weights=[1/n_gram]*n_gram, smoothing_function=smoothing)
        bleu_scores.append(score)
    return sum(bleu_scores) / len(bleu_scores)

def get_token_length(tokenizer, prediction):
    return len(tokenizer.encode(prediction))