from transformers import set_seed
from vllm import LLM, SamplingParams
from dataclasses import dataclass, field
from typing import List
from collections import Counter
from transformers import AutoTokenizer

@dataclass
class GenerationArguments:
    model_name: str = field(default="gpt2", metadata={"help": "The model name"})
    temperature: float = field(default=0.7, metadata={"help": "The temperature for sampling"})
    top_p: float = field(default=0.99, metadata={"help": "The nucleus sampling top_p"})
    top_k: int = field(default=50, metadata={"help": "The nucleus sampling top_k"})
    num_return_sequences: int = field(default=20, metadata={"help": "The number of return sequences"})
    seed: int = field(default=1, metadata={"help": "The random seed"})
    max_new_tokens: int = field(default=256, metadata={"help": "The maximum number of new tokens to generate"})
    batch_size: int = field(default=16, metadata={"help": "The batch size for generation"})
    tensor_parallel_size: int = field(default=2, metadata={"help": "The tensor parallel size"})



class vLLMWrapperForCompletionModel:
    def __init__(self, generation_args):
        self.model = LLM(
            model=generation_args.model_name,
            tokenizer=generation_args.model_name,
            tensor_parallel_size=generation_args.tensor_parallel_size,
            max_model_len=2048
        )
        self.generation_args = generation_args
        set_seed(self.generation_args.seed)

        self.sampling_params = SamplingParams(
            temperature=self.generation_args.temperature,
            top_p=self.generation_args.top_p,
            top_k=self.generation_args.top_k,
            n=self.generation_args.num_return_sequences,
            max_tokens=self.generation_args.max_new_tokens,
        )
        self.greedy_samp_params = SamplingParams(
            best_of=1, temperature=0.0, top_p=1, top_k=-1, use_beam_search=False, max_tokens=self.generation_args.max_new_tokens
        )
    
    def prepare_prompts(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        return prompts
    
    def greedy_generate(self, prompts):
        prompts = self.prepare_prompts(prompts)
        greedy_request_output = self.model.generate(prompts, sampling_params=self.greedy_samp_params, use_tqdm=False)
        return [one_request.outputs[0].text for one_request in greedy_request_output]
    
    def sampling_generate(self, prompts) -> List[Counter]:
        prompts = self.prepare_prompts(prompts)
        samp_request_output = self.model.generate(prompts=prompts, sampling_params=self.sampling_params, use_tqdm=False)
        samp_texts = [[cpl.text for cpl in one_request.outputs if cpl.text.strip()] for one_request in samp_request_output]
        samp_texts_counter = [Counter(texts) for texts in samp_texts]
        return samp_texts_counter


class vLLMWrapperForChatModel(vLLMWrapperForCompletionModel):
    def __init__(self, generation_args):
        super().__init__(generation_args)
        self.chat_template_fn = self.get_chat_template_fn(generation_args.model_name)

    def get_chat_template_fn(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        chat_template_fn = lambda prompts: [
            tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
            for prompt in prompts
        ]
        return chat_template_fn
    
    def prepare_prompts(self, prompts):
        prompts = super().prepare_prompts(prompts)
        return self.chat_template_fn(prompts)
