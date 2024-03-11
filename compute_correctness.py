from metrics import rouge_and_similarity_score
import json
import argparse
import os
import pandas as pd
from tqdm import tqdm

ROUGE_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.9

def compute_correctness(df):
    ground_truths, greedy_responses = [], []
    for i, d in df.iterrows():
        ground_truths.extend(d["ground_truth"])
        greedy_responses.extend([d["greedy_response"]]*len(d["ground_truth"])) # compute correctness for each ground truth
    rouge_scores, cosine_similarity = rouge_and_similarity_score(ground_truths, greedy_responses)
    correctness_iter = iter(zip(rouge_scores, cosine_similarity))
    for i, d in df.iterrows():
        best_rouge, best_rouge_match, best_similarity, best_similarity_match = 0, "", 0, ""
        for gt in d["ground_truth"]:
            rouge, similarity = next(correctness_iter)
            if rouge > best_rouge:
                best_rouge, best_rouge_match = rouge, gt
            if similarity > best_similarity:
                best_similarity, best_similarity_match = similarity, gt
        df.loc[i, "rouge"] = best_rouge
        df.loc[i, "rouge_correctness"] = best_rouge > ROUGE_THRESHOLD
        df.loc[i, "best_rouge_match"] = best_rouge_match
        df.loc[i, "cosine_similarity"] = best_similarity
        df.loc[i, "similarity_correctness"] = best_similarity > SIMILARITY_THRESHOLD
        df.loc[i, "best_similariy_match"] = best_similarity_match
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    dataset_names = ['coqa', 'nq', 'squad', 'triviaqa']
    input_file_suffix = "_responses.json"
    output_file_suffix = "_correctness.json"

    for dataset_name in dataset_names:
        input_path = os.path.join(args.data_dir, f"{dataset_name}{input_file_suffix}")
        output_path = os.path.join(args.data_dir, f"{dataset_name}{output_file_suffix}")
        response_data = pd.DataFrame(json.load(open(input_path)))
        print(f"Read {len(response_data)} responses from {input_path}")
        response_data = compute_correctness(response_data)
        response_data.to_json(output_path, orient="records", lines=True)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
