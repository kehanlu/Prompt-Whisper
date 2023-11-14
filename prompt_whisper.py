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
# import cusomize code
from model import MyWhisperForConditionalGeneration

cc = OpenCC('s2t')

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--language', '-l', type=str)
    parser.add_argument('--prompt', '-p', type=str)
    parser.add_argument('--model_name_or_path', '-m', type=str, default="openai/whisper-large-v3")
    parser.add_argument('--output_dir', '-o', type=str, default=".")
    parser.add_argument('--exp_name', '-n', type=str, default="results")
    parser.add_argument('--overwrite_forced_decoder_ids', '-c', type=str)
    
    return parser.parse_args()

def insert_space_in_code_switched_text(text):
    text = text.lower()
    # Regular expression to match Chinese characters
    chinese_char_pattern = r'[\u4e00-\u9fff]'

    # Insert space before and after each Chinese character
    spaced_text = re.sub(f'({chinese_char_pattern})', r' \1 ', text)

    # Remove any extra spaces added by the previous step
    normalized_text = re.sub(r'\s+', ' ', spaced_text)
    normalized_text = normalized_text.strip().replace("  ", " ")
    return normalized_text

def calculate_MER(results):
    hyps = []
    refs = []
    new_results = []
    for result in results:
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
    # Load dataset
    DATASET_PATH = "chiyuanhsiao/ML2021_HungyiLee_Corpus"

    dataset = load_dataset(DATASET_PATH, split="test").shuffle(42)[:3000]
    dataset = Dataset.from_dict(dataset) # DO NOT MODIFY
    print("="*15, "Dataset Info", "="*15)
    print("Dataset:", DATASET_PATH)
    print(dataset)

    # Load model
    print("="*15, "Model Info", "="*15)
    device = "cuda"
    model_name_or_path = args.model_name_or_path
    model = MyWhisperForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir="/home/u2619111/hank/cache/").to(device)
    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    print("Model:", model_name_or_path)
    
    
    # Create dataloader
    def collate_fn(batch):
        file = [item['file'] for item in batch]
        audio = [item['audio']["array"] for item in batch]
        transcription = [item['transcription'] for item in batch]
        inputs = {
            "file": file,
            "audio": processor(audio, sampling_rate=16000, return_tensors="pt").input_features,
            "transcription": transcription
        }
        return inputs

    batch_size = 32
    dataloader = DataLoader(
        dataset,  
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    # Start Inference
    model.eval()
    prompt = args.prompt
    prompt_ids = processor.get_prompt_ids(prompt) if prompt else None
    
    if args.overwrite_forced_decoder_ids is not None:
        overwrite_forced_decoder_ids = []
        token_ids = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize(args.overwrite_forced_decoder_ids))
        for i, token_id in enumerate(token_ids):
            overwrite_forced_decoder_ids.append((i+1, token_id))
    else:
        overwrite_forced_decoder_ids = None

    generation_config = model.generation_config
    generate_options = {
        "language": args.language,
        "prompt_ids": prompt_ids,
        "task": args.task,
        "overwrite_force_decoder_ids": overwrite_forced_decoder_ids,
    }

    print("="*15, "Inference Info", "="*15)
    print("batch_size:", batch_size)
    print("generate_options:", generate_options)

    results = []
    for b in tqdm(dataloader):
        generated_ids, generation_config = model.generate(
            inputs=b["audio"].to(device),
            generation_config=generation_config,
            **generate_options
        )
        
        predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for file, t, p in zip(b["file"], b["transcription"], predictions):
            if prompt is not None:
                p = p.replace(" "+prompt, "", 1) # remove prompt
            results.append({
                "id": file,
                "prediction": p,
                "transcription": t
            })

    print("forced_decoder_ids:", processor.decode([y for x,y in generation_config.forced_decoder_ids])) # for debug
    # Calculate MER
    results, word_error_rate = calculate_MER(results)
    print("Number of data:", len(results))
    print("WER:", word_error_rate)


    generate_options["prompt_ids"] = generate_options["prompt_ids"].tolist() if prompt is not None else None
    json.dump(
        {"MER": word_error_rate, "generate_options": generate_options,"results": results},open(f"{args.output_dir}/{args.exp_name}.json", "w"), indent=2, ensure_ascii=False
    )
    print(f"Output file: {args.output_dir}/{args.exp_name}.json")


if __name__ == "__main__":
    args = get_args_parser()
    main(args=args)