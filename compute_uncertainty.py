import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = "false"

ALPHA_FOR_EIGEN_SCORE = 1e-3

import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from collections import Counter

from rouge_score.rouge_scorer import RougeScorer
rouge_scorer = RougeScorer(['rougeL'], use_stemmer=True)
rouge_fn = lambda target, pred: rouge_scorer.score(target, pred)['rougeL'].fmeasure

def prepare_inputs(prompt, generations, tokenizer):
    prompt_encoded = tokenizer(prompt, return_tensors="pt", padding=True)
    full_sentences = [f"{prompt}{generation}{tokenizer.eos_token}" for generation in generations] # add eos token is for eigen score
    full_sentences_encoded = tokenizer(full_sentences, return_tensors="pt", padding=True)
    return prompt_encoded, full_sentences_encoded

def mask_logits_for_perplexity(logits, attention_mask):
    # eos tokens should be ignored when computing perplexity
    extended_mask = torch.cat([attention_mask, torch.zeros(attention_mask.shape[0], 1, dtype=torch.long)], dim=1)
    shifted_attention_mask = extended_mask[:, 1:]
    return logits.masked_fill(~shifted_attention_mask.bool().unsqueeze(-1), float('nan'))


def compute_logits_and_hidden_states(prompt_encoded, full_sentences_encoded, model, layer=15):
    prompt_length = prompt_encoded['input_ids'].shape[1]
    batch_size = full_sentences_encoded['input_ids'].shape[0]
    with torch.no_grad():
        prompt_output = model(**prompt_encoded)
        batched_prompt_pkv = [(k.repeat(batch_size, 1, 1, 1), v.repeat(batch_size, 1, 1,1)) for k, v in prompt_output.past_key_values]
        full_sentences_encoded['input_ids'] = full_sentences_encoded['input_ids'][:, prompt_length:]
        full_sentences_output = model(**full_sentences_encoded, past_key_values=batched_prompt_pkv, output_hidden_states=True)
    # slice the attention_mask to remove the prompt
    attention_mask = full_sentences_encoded['attention_mask'][:, prompt_length:]
    answer_logits =  torch.cat(
        [prompt_output.logits[:,-1,:].repeat(batch_size, 1, 1), full_sentences_output.logits[:, :-1, :]], 
        dim=1
    )
    answer_logits = mask_logits_for_perplexity(answer_logits, attention_mask)
    # select the eos hidden states
    eos_indices = attention_mask.sum(dim=1) - 1
    eos_hidden_states = full_sentences_output.hidden_states[layer][range(batch_size), eos_indices]
    return answer_logits, eos_hidden_states
    
def compute_perplexities(logits: torch.tensor, input_ids: torch.tensor):
    assert logits.shape[:2] == input_ids.shape
    log_probs = F.log_softmax(logits, dim=-1)
    target_prob = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    return torch.nanmean(-target_prob, dim=1).tolist()

def compute_energy_score(logits):
    log_sum_exp = torch.logsumexp(logits, dim=-1)
    return torch.nanmean(log_sum_exp, dim=1).tolist()

def compute_eigen_score(z:torch.tensor, alpha: float=ALPHA_FOR_EIGEN_SCORE):
    k,d = z.shape
    j_d = torch.eye(d) - (1/d) * torch.ones(d, d)
    sigma = torch.einsum('ij,jk,kl->il', z, j_d, z.t()) # 原论文里的z是d*K的，这里是K*d的，所以这里的z.t()是d*K的
    return ((1/k) * torch.logdet(sigma + alpha * torch.eye(k))).item()

def manual_check_sentence(sentence):
    if not sentence.strip():
        return False
    if "Q: " in sentence:
        return sentence.split("Q: ")[0]
    else:
        return sentence

def compute_uncertainty(df, model, tokenizer, dataset_name):
    
    total_prompts = df['prompt']
    common_prefix = os.path.commonprefix(total_prompts) # find the common (instruction+example) prefix
    common_prefix_encoded = tokenizer(common_prefix, return_tensors="pt")
    common_prefix_kv = model(**common_prefix_encoded)['past_key_values']


    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Calculating uncertainty for {dataset_name}"):   
        prompt_encoded, full_sentences_encoded = prepare_inputs(row['prompt'], [row['greedy_response']]+list(row['sampling_response'].keys()), tokenizer)
        answer_logits, eos_hidden_states = compute_logits_and_hidden_states(prompt_encoded, full_sentences_encoded, model)
        perplexities = compute_perplexities(answer_logits, full_sentences_encoded['input_ids'])
        if not row['greedy_response']: # if the greedy response is empty, set the perplexity to a very high value
            perplexities[0] = 1000 
        assert not any(np.isnan(perplexities))
        
        weights = list(row['sampling_response'].values())
        # logits based metrics
        perplexity = perplexities[0]
        ln_entropy = np.average(perplexities[1:], weights=weights)
        energy_score = compute_energy_score(answer_logits)
        energy_score = np.average(energy_score[1:], weights=weights)
        # hidden states based metrics
        eos_hidden_states = eos_hidden_states[1:, :] # remove the greedy generation
        eos_hidden_states = torch.cat(
            [eos_hidden_states[i].repeat(weights[i], 1) for i in range(len(weights))], 
            dim=0
        )
        eigen_score = compute_eigen_score(eos_hidden_states)

        df.at[i, 'perplexity'] = perplexity
        df.at[i, 'ln_entropy'] = ln_entropy
        df.at[i, 'energy_score'] = energy_score
        df.at[i, 'eigen_score'] = eigen_score
    return df

def compute_similarity_for_row(row, rouge_fn):
    sentences = list(row['sampling_response'].keys())
    sentence_pair_counter = Counter()
    same_pair_cnt = 0
    for i, sentence in enumerate(sentences):
        same_pair_cnt += row['sampling_response'][sentence] * (row['sampling_response'][sentence] - 1) // 2
        for j in range(i+1, len(sentences)):
            sentence_pair_counter[(sentence, sentences[j])] = row['sampling_response'][sentence] * row['sampling_response'][sentences[j]]
    sentence_pair_counter['same'] = same_pair_cnt
    total_sims = total_pairs = 0
    for pair, cnt in sentence_pair_counter.items():
        similarity = rouge_fn(pair[0], pair[1]) if pair != 'same' else 1
        total_sims += similarity * cnt
        total_pairs += cnt
    return total_sims / total_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/data/MODELS/Mistral-7B-v0.1", help="The model name")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    file_suffix = "_correctness.json"

    for dataset_name in ["coqa", "triviaqa", "squad", "nq"]:
        file_path = os.path.join(args.data_dir, f"{dataset_name}{file_suffix}")
        df = pd.read_json(file_path, orient="records", lines=True)
        
        df = compute_uncertainty(df, model, tokenizer, dataset_name)

        tqdm.pandas(desc=f"Calculating lexical similarity for {dataset_name}")
        df['lexical_similarity'] = df.progress_apply(compute_similarity_for_row, axis=1, rouge_fn=rouge_fn)

        # remove unnecessary columns
        if 'context' in df.columns:
            df = df.drop(columns=["context"])
        df = df.drop(columns=["sampling_response", "prompt", "greedy_response", "best_rouge_match", "best_similariy_match"])

        save_path = os.path.join(args.data_dir, f"{dataset_name}_uncertainty.jsonl")
        df.to_json(save_path, lines=True, orient="records")
        print(f"Finished {dataset_name} to {save_path}")


if __name__=="__main__":
    main()