import os
os.environ["HTTP_PROXY"]="http://127.0.0.1:7898"
os.environ["HTTPS_PROXY"]="http://127.0.0.1:7898"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import HfArgumentParser
from models import GenerationArguments, vLLMWrapperForChatModel, vLLMWrapperForCompletionModel
from tqdm import tqdm
import json
from dataclasses import dataclass, field

@dataclass
class MyGenerationArguments(GenerationArguments):
    template_path: str = field(default="./templates/zero_shot.json", metadata={"help": "The path to the template file"})
    mode: str = field(default="full", metadata={"help": "Debug(100 samples) or full dataset"})
    model_type: str = field(default="chat", metadata={"help": "The type of model to use for generation"})

def generate_response(model, data, batch_size, template_fn, output_file):
    p_bar = tqdm(total=len(data))
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        prompts = []
        for example in batch:
            template_input = example.copy()
            template_input.pop('id')
            template_input.pop('ground_truth')
            prompts.append(template_fn(**template_input))
        greedy_responses = model.greedy_generate(prompts)
        sampling_responses = model.sampling_generate(prompts)
        for j, example in enumerate(batch):
            example['prompt'] = prompts[j]
            example['greedy_response'] = greedy_responses[j]
            example['sampling_response'] = sampling_responses[j]
            json.dump(example, output_file, ensure_ascii=False, indent=4)
            output_file.write(",\n")
            p_bar.update(1)
    

def main():
    parser = HfArgumentParser(MyGenerationArguments)
    generation_args = parser.parse_args_into_dataclasses()[0]
    batch_size = generation_args.batch_size

    print(f"Generation Arguments: {generation_args}")
    if generation_args.model_type == "chat":
        model = vLLMWrapperForChatModel(generation_args)
    elif generation_args.model_type == "completion":
        model = vLLMWrapperForCompletionModel(generation_args)
    else:
        raise ValueError(f"Invalid model type: {generation_args.model_type}")

    if not os.path.exists("./outputs/debug"):
        os.makedirs("./outputs/debug")

    for ds_name in ['triviaqa']:
        if generation_args.mode == "debug":
            ds_path = f"./data/debug/{ds_name}_sample.json"
            print(f"Debug mode: Generating responses for {ds_name} dataset")
        else:
            ds_path = f"./data/processed/{ds_name}.json"
            print(f"Generating responses for full {ds_name} dataset")
        data = json.load(open(ds_path))
        templates = json.load(open(generation_args.template_path))
        if 'context' in data[0]:
            template_fn = lambda question, context: '\n'.join(templates['with_context']).format(question=question, context=context)
        else:
            template_fn = lambda question: '\n'.join(templates['without_context']).format(question=question)
        print(f"Generating responses for {ds_name} dataset")
        with open(f"./outputs/debug/{ds_name}_responses.json", "w") as output_file:
            output_file.write("[\n")
            generate_response(model, data, batch_size, template_fn, output_file)
            output_file.seek(output_file.tell()-2)
            output_file.write("\n]")
            
if __name__ == "__main__":
    main()