# Prompt-Whisper

## Setup

```sh
conda create --prefix conda/whisper python=3.10

pip install openai-whisper datasets transformers librosa soundfile opencc-python-reimplemented jiwer
```

## Prompt Whisper

We random sample 3000 utterances from [chiyuanhsiao/ML2021_HungyiLee_Corpus](https://huggingface.co/datasets/chiyuanhsiao/ML2021_HungyiLee_Corpus) in this script.

- `--model_name_or_path`, `-m`: This parameter allows you to specify the Whisper model you want to use. For example, you can use models like `openai/whisper-large-v3` or `openai/whisper-base`.


- Generation Options: You have the flexibility to customize the generation process using several options. Refer to the [transformers.WhisperForConditionalGeneration.generate](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/whisper#transformers.WhisperForConditionalGeneration) function for more details. These options include:
    - `--task`, `-t`: Specify the task you want the model to perform, which can be either `transcribe` or `translate`.
    - `--language`, `-l`: Provide the language tag for the input or output text. For instance, you can use language codes like `zh` for Chinese or `en` for English.
    - `--prompt`, `-p`: Input your prompt text.
- `--overwrite_forced_decoder_ids`, `-c`: This option allows you to override the `force_decoder_id` within the `generate()` function. This customization gives you greater control over the model's behavior during generation.
- `--output_dir`: Path for the results file.

```sh
python prompt_whisper.py -t transcribe -l zh -m "openai/whisper-base"

python prompt_whisper.py -t transcribe -l zh -p "我喜歡機器學習"

python prompt_whisper.py -p "真是太厲害了"

python prompt_whisper.py -c "<|en|><|zh|><|transcribe|><|notimestamps|>"

python prompt_whisper.py -t transcribe -l zh -c "<|en|><|zh|><|transcribe|><|notimestamps|>" -p "這是一個機器學習的例子" 
```

### Error Rate

To determine the mixed error rate, we will follow this procedure:

- Convert simplified Chinese characters to traditional Chinese characters.
- Insert spaces between Chinese characters and English words

Example:
```json
[{
    "id": "0_1891_1894.mp3",
    "prediction": "我 們 不 止 訓 練 一 個 classifier 來 解 任 務 一",
    "transcription": "我 們 不 止 訓 練 一 個 classifier 來 解 任 務 一",
    "raw_prediction": "我们不止训练一个classifier来解任务一"
},
{
    "id": "5_1722_1725.mp3",
    "prediction": "這 個 tensor 的 大 小 是 5 乘 以 10 乘 以 3",
    "transcription": "這 個 tensor 的 大 小 是 5 乘 以 10 乘 以 3",
    "raw_prediction": "這個 tensor的大小是5乘以10乘以3"
},
{
    "id": "6_1153_1156.mp3",
    "prediction": "是 要 把 source domain 跟 target domain 分 開",
    "transcription": "是 要 把 source domain 跟 target domain 分 開",
    "raw_prediction": "是要把source domain跟target domain分開"
}]
```

`raw_prediction` represents the original output sequence from whisper.