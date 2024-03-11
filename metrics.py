from rouge_score.rouge_scorer import RougeScorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances

def rouge_and_similarity_score(targets, preds):
    rouge_scorer = RougeScorer(['rougeL'], use_stemmer=True)
    se_model = SentenceTransformer('sentence-transformers/nli-roberta-large')
    rouge_scores = []
    for t, p in zip(targets, preds):
        rouge_scores.append(rouge_scorer.score(t, p)['rougeL'].fmeasure)
    target_embeddings = se_model.encode(targets, show_progress_bar=False)
    prediction_embeddings = se_model.encode(preds, show_progress_bar=False)
    cosine_similarity = 1 - paired_cosine_distances(target_embeddings, prediction_embeddings)
    return rouge_scores, cosine_similarity