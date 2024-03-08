import os
os.environ["HTTP_PROXY"]="http://127.0.0.1:7898"
os.environ["HTTPS_PROXY"]="http://127.0.0.1:7898"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import HfArgumentParser
from models import GenerationArguments, get_inference_model
from tqdm import tqdm
import json
from dataclasses import dataclass, field

import wandb
import time

@dataclass
class MyGenerationArguments(GenerationArguments):
    template_path: str = field(default="./templates/zero_shot.json", metadata={"help": "The path to the template file"})
    is_debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode(only 100 sample from each dataset)"})
    model_type: str = field(default="chat", metadata={"help": "The type of model to use for generation"})
    run_name: str = field(default="response_generation", metadata={"help": "The name of the run"})
    context_stop_sign: str = field(default=None, metadata={"help": "The stop token for context"})
    question_stop_sign: str = field(default=None, metadata={"help": "The stop token for question"})

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
    return example # only for logging
    
def create_output_dir(run_name):
    output_dir = f"./outputs/{run_name}"
    output_dir += f"_{time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()+3600*8))}" # UTC+8
    unqiue_id = 0
    while os.path.exists(output_dir) and os.listdir(output_dir): # if exists and not empty
        log_dir += f"_{unqiue_id}"
        unqiue_id += 1
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    return output_dir

def load_template(template_path):
    templates = json.load(open(template_path))
    prompt_artifact = wandb.Artifact(name="template", type="prompt")
    prompt_artifact.add_file(template_path)
    wandb.log_artifact(prompt_artifact)
    context_template_fn = lambda question, context: '\n'.join(templates['with_context']).format(question=question, context=context)
    no_context_template_fn = lambda question: '\n'.join(templates['without_context']).format(question=question)
    return context_template_fn, no_context_template_fn

def main():
    parser = HfArgumentParser(MyGenerationArguments)
    generation_args = parser.parse_args_into_dataclasses()[0]
    batch_size = generation_args.batch_size

    wandb.init(project="HallucinationDetection", name=generation_args.run_name, config=generation_args)
    print(f"Generation Arguments: {generation_args}")
    
    output_dir = create_output_dir(generation_args.run_name)
    context_template_fn, no_context_template_fn = load_template(generation_args.template_path)
    
    # we log one example for each dataset to better understand the generated responses
    log_columns = ["id", "prompt", "greedy_response", "ground_truth"]
    generate_example_table = wandb.Table(columns=log_columns)
    wandb.config['output'] = []

    model = get_inference_model(generation_args)

    for ds_name in ["coqa", "nq", "squad", "triviaqa"]:
        if generation_args.is_debug:
            ds_path = f"./data/debug/{ds_name}_sample.json"
            print(f"Debug mode: Using sampled {ds_name} dataset")
        else:
            ds_path = f"./data/processed/{ds_name}.json"
            print(f"Full mode: Using full {ds_name} dataset")
        data = json.load(open(ds_path))
        
        if 'context' in data[0]:
            template_fn = context_template_fn
            model.generation_args.stop_sign = generation_args.context_stop_sign
        else:
            template_fn = no_context_template_fn
            model.generation_args.stop_sign = generation_args.question_stop_sign
        model.update_sampling_params()

        print(f"Generating responses for {ds_name} dataset")
        output_path = os.path.join(output_dir, f"{ds_name}_responses.json")
        with open(output_path, "w") as output_file:
            output_file.write("[\n")
            last_example = generate_response(model, data, batch_size, template_fn, output_file)
            output_file.seek(output_file.tell()-2)
            output_file.write("\n]")

        # log last example
        generate_example_table.add_data(*[str(last_example[k]) for k in log_columns])
        wandb.config['output'].append(output_path)
        
    wandb.log({"generated_responses": generate_example_table})
    wandb.finish()
    return 0
            
if __name__ == "__main__":
    print(main())