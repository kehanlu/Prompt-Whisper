import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import WhisperProcessor
import argparse
from opencc import OpenCC
from jiwer import wer
import json
from tqdm import tqdm
import re
from model import MyWhisperForConditionalGeneration

# Creating an instance of OpenCC for Simplified to Traditional Chinese conversion.
cc = OpenCC('s2t')

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-d', type=str, default="chiyuanhsiao/ML2021_HungyiLee_Corpus")
    parser.add_argument('--device', '-v', type=str, default="cuda")
    parser.add_argument('--cache_dir', '-s', type=str, default="./")
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--language', '-l', type=str)
    parser.add_argument('--prompt', '-p', type=str)
    parser.add_argument('--model_name_or_path', '-m', type=str, default="openai/whisper-large-v3")
    parser.add_argument('--output_dir', '-o', type=str, default="./results")
    parser.add_argument('--exp_name', '-n', type=str, default="results")
    parser.add_argument('--overwrite_forced_decoder_ids', '-c', type=str)
    
    return parser.parse_args()


def insert_space_in_code_switched_text(text):
    text = text.lower()
    # Regular expression to match Chinese characters.
    chinese_char_pattern = r'[\u4e00-\u9fff]'

    # Insert space before and after each Chinese character.
    spaced_text = re.sub(f'({chinese_char_pattern})', r' \1 ', text)

    # Remove any extra spaces added by the previous step.
    normalized_text = re.sub(r'\s+', ' ', spaced_text)
    normalized_text = normalized_text.strip().replace("  ", " ")
    return normalized_text


def calculate_MER(results):
    hyps = []
    refs = []
    new_results = []

    for result in results:
        # Convert traditional chinese to simplified chinese.
        p = cc.convert(result["prediction"])

        p = insert_space_in_code_switched_text(p)
        hyps.append(p)

        refs.append(result["transcription"])

        new_results.append({
            "id": result["id"],
            "prediction": p,
            "transcription": result["transcription"],
            "raw_prediction": result["prediction"],
        })

    return new_results, wer(hyps, refs)


def main(args):

    # Load dataset from the specified path in the arguments.
    DATASET_PATH = args.dataset_path

    # Cache dir.
    cache_dir = args.cache_dir

    ### ------ DO NOT MODIFY "dataset_index". ------ ##
    dataset_index = 5642
    dataset = load_dataset(DATASET_PATH, split="test", cache_dir=cache_dir)[:dataset_index]
    ### -------------------------------------------- ##

    # Convert the dataset to a dictionary format.
    dataset = Dataset.from_dict(dataset) 

    # Printing information about the dataset.
    print("="*15, "Dataset Info", "="*15)
    print("Dataset:", DATASET_PATH)

    # Printing information about the model.
    print("="*15, "Model Info", "="*15)

    # Specifying the device to use.
    device = args.device

    # Load the model and the processor.
    model_name_or_path = args.model_name_or_path
    model = MyWhisperForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir="/home/u2619111/hank/cache/").to(device)
    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    print("Model:", model_name_or_path)
    
    # Function to collate data into batches.
    def collate_fn(batch):

        # Extracting file names, audio data, and transcriptions from the batch.
        file = [item['file'] for item in batch]
        audio = [item['audio']["array"] for item in batch]
        transcription = [item['transcription'] for item in batch]

        # Preparing input data for the model.
        inputs = {
            "file": file,
            "audio": processor(audio, sampling_rate=16000, return_tensors="pt").input_features,
            "transcription": transcription
        }
        return inputs

    batch_size = args.batch_size

    # Creating a DataLoader for batch processing.
    dataloader = DataLoader(
        dataset,  
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    # Preparing the model for inference (evaluation mode).
    model.eval()

    # Processing the prompt if provided.
    prompt = args.prompt
    prompt_ids = processor.get_prompt_ids(prompt) if prompt else None

    # Overwrite forced decoder IDs if specified.    
    if args.overwrite_forced_decoder_ids is not None:
        overwrite_forced_decoder_ids = []
        token_ids = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize(args.overwrite_forced_decoder_ids))
        for i, token_id in enumerate(token_ids):
            overwrite_forced_decoder_ids.append((i+1, token_id))
    else:
        overwrite_forced_decoder_ids = None

    # Configuration for generating predictions.
    generation_config = model.generation_config
    generate_options = {
        "language": args.language,
        "task": args.task,
        "prompt_ids": prompt_ids,
        "overwrite_force_decoder_ids": overwrite_forced_decoder_ids,
    }

    # Printing information about the inference.
    print("="*15, "Inference Info", "="*15)
    print("batch_size:", batch_size)
    print("generate_options:", generate_options)

    # Iterating over the dataset and generating predictions.
    results = []
    for b in tqdm(dataloader):
        generated_ids, generation_config = model.generate(
            inputs=b["audio"].to(device),
            generation_config=generation_config,
            **generate_options
        )
        
        # Decoding generated IDs to text.
        predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Storing results and removing the prompt text if present.
        for file, t, p in zip(b["file"], b["transcription"], predictions):
            if prompt is not None:
                p = p.replace(" "+prompt, "", 1) # remove prompt
            results.append({
                "id": file,
                "prediction": p,
                "transcription": t
            })

    # if generation_config.forced_decoder_ids is not None:
    #     print(generation_config.forced_decoder_ids)
    #     print("forced_decoder_ids:", processor.decode([y for x,y in generation_config.forced_decoder_ids])) # for debug
    
    # Calculate Word Error Rate (WER) for the results.
    results, word_error_rate = calculate_MER(results)
    print("Number of data:", len(results))
    print("WER:", word_error_rate)

    # Preparing generate_options for saving.
    generate_options["prompt_ids"] = generate_options["prompt_ids"].tolist() if prompt is not None else None
    
    # Saving the results to a JSON file.
    json.dump(
        {"model_name_or_path": model_name_or_path, "MER": word_error_rate, "generate_options": generate_options,"results": results},open(f"{args.output_dir}/{args.exp_name}.json", "w"), indent=2, ensure_ascii=False
    )
    print(f"Output file: {args.output_dir}/{args.exp_name}.json")


if __name__ == "__main__":
    args = get_args_parser()
    main(args=args)