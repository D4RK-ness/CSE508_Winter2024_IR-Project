import nltk
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

# Function to compute BLEU score
def compute_bleu(reference, hypothesis):
    return corpus_bleu([[ref.split()] for ref in reference], [hyp.split() for hyp in hypothesis])

# Function to compute ROUGE scores
def compute_rouge(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']

# Function to evaluate models
def evaluate_models(models, reference_texts, generated_texts):
    results = {}
    for model_name, gen_texts in generated_texts.items():
        bleu_scores = []
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        for i, gen_text in enumerate(gen_texts):
            bleu = compute_bleu(reference_texts[i], [gen_text])
            rouge_1, rouge_2, rouge_l = compute_rouge(reference_texts[i], gen_text)
            bleu_scores.append(bleu)
            rouge_1_scores.append(rouge_1)
            rouge_2_scores.append(rouge_2)
            rouge_l_scores.append(rouge_l)
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores)
        avg_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores)
        avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
        results[model_name] = {
            'BLEU': avg_bleu,
            'ROUGE-1': avg_rouge_1,
            'ROUGE-2': avg_rouge_2,
            'ROUGE-L': avg_rouge_l
        }
    return results

# Example usage
reference_texts = ["reference text 1", "reference text 2", "reference text 3"]
generated_texts = {
    "Model 1": ["generated text 1 for model 1", "generated text 2 for model 1", "generated text 3 for model 1"],
    "Model 2": ["generated text 1 for model 2", "generated text 2 for model 2", "generated text 3 for model 2"]
}

evaluation_results = evaluate_models(["Model 1", "Model 2"], reference_texts, generated_texts)
print(evaluation_results)
