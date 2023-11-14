import torch
from transformers import WhisperForConditionalGeneration
from typing import Optional, Tuple, Union
from transformers.utils import logging
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE, TASK_IDS

logger = logging.get_logger(__name__)


class MyWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.post_init()

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        return_timestamps=None,
        task=None,
        language=None,
        is_multilingual=None,
        prompt_ids: Optional[torch.Tensor] = None,
        return_token_timestamps=None,
        overwrite_force_decoder_ids=None,
        **kwargs,
    ):
        """

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            return_timestamps (`bool`, *optional*):
                Whether to return the timestamps with the text. This enables the `WhisperTimestampsLogitsProcessor`.
            task (`str`, *optional*):
                Task to use for generation, either "translate" or "transcribe". The `model.config.forced_decoder_ids`
                will be updated accordingly.
            language (`str`, *optional*):
                Language token to use for generation, can be either in the form of `<|en|>`, `en` or `english`. You can
                find all the possible language tokens in the `model.generation_config.lang_to_id` dictionary.
            is_multilingual (`bool`, *optional*):
                Whether or not the model is multilingual.
            prompt_ids (`torch.Tensor`, *optional*):
                Rank-1 tensor of token IDs created by passing text to [`~WhisperProcessor.get_prompt_ids`] that is
                provided as a prompt to each chunk. This can be used to provide or "prompt-engineer" a context for
                transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those words
                correctly. It cannot be used in conjunction with `decoder_start_token_id` as it overwrites this value.
            return_token_timestamps (`bool`, *optional*):
                Whether to return token-level timestamps with the text. This can be used with or without the
                `return_timestamps` option. To get word-level timestamps, use the tokenizer to group the tokens into
                words.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchDecoderOnlyOutput`],
                    - [`~generation.SampleDecoderOnlyOutput`],
                    - [`~generation.BeamSearchDecoderOnlyOutput`],
                    - [`~generation.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchEncoderDecoderOutput`],
                    - [`~generation.SampleEncoderDecoderOutput`],
                    - [`~generation.BeamSearchEncoderDecoderOutput`],
                    - [`~generation.BeamSampleEncoderDecoderOutput`]
        """
        if generation_config is None:
            generation_config = self.generation_config

        if return_timestamps is not None:
            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You are trying to return timestamps, but the generation config is not properly set."
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`."
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                )

            generation_config.return_timestamps = return_timestamps
        else:
            generation_config.return_timestamps = False

        if language is not None:
            if not hasattr(generation_config, "lang_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `language` argument"
                    "to `generate`. Either set the language using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            language = language.lower()
            generation_config.language = language
        if task is not None:
            if not hasattr(generation_config, "task_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `task` argument"
                    "to `generate`. Either set the task using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.task = task

        forced_decoder_ids = None

        # Legacy code for backward compatibility
        if hasattr(self.config, "forced_decoder_ids") and self.config.forced_decoder_ids is not None:
            forced_decoder_ids = self.config.forced_decoder_ids
        elif (
            hasattr(self.generation_config, "forced_decoder_ids")
            and self.generation_config.forced_decoder_ids is not None
        ):
            forced_decoder_ids = self.generation_config.forced_decoder_ids
        else:
            forced_decoder_ids = kwargs.get("forced_decoder_ids", None)

        if task is not None or language is not None or (forced_decoder_ids is None and prompt_ids is not None):
            forced_decoder_ids = []
            if hasattr(generation_config, "language"):
                if generation_config.language in generation_config.lang_to_id.keys():
                    language_token = generation_config.language
                elif generation_config.language in TO_LANGUAGE_CODE.keys():
                    language_token = f"<|{TO_LANGUAGE_CODE[generation_config.language]}|>"
                elif generation_config.language in TO_LANGUAGE_CODE.values():
                    language_token = f"<|{generation_config.language}|>"
                else:
                    is_language_code = len(generation_config.language) == 2
                    raise ValueError(
                        f"Unsupported language: {generation_config.language}. Language should be one of:"
                        f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                    )
                forced_decoder_ids.append((1, generation_config.lang_to_id[language_token]))
            else:
                forced_decoder_ids.append((1, None))  # automatically detect the language

            if hasattr(generation_config, "task"):
                if generation_config.task in TASK_IDS:
                    forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
                else:
                    raise ValueError(
                        f"The `{generation_config.task}`task is not supported. The task should be one of `{TASK_IDS}`"
                    )
            elif hasattr(generation_config, "task_to_id"):
                forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))  # defaults to transcribe
            if hasattr(generation_config, "no_timestamps_token_id") and not generation_config.return_timestamps:
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

        if forced_decoder_ids is not None:
            generation_config.forced_decoder_ids = forced_decoder_ids

            if overwrite_force_decoder_ids is not None:
                # hank
                generation_config.forced_decoder_ids = overwrite_force_decoder_ids

        if prompt_ids is not None:
            if kwargs.get("decoder_start_token_id") is not None:
                raise ValueError(
                    "When specifying `prompt_ids`, you cannot also specify `decoder_start_token_id` as it gets overwritten."
                )
            prompt_ids = prompt_ids.tolist()
            decoder_start_token_id, *text_prompt_ids = prompt_ids
            # Slicing the text prompt ids in a manner consistent with the OpenAI implementation
            # to accomodate context space for the prefix (see https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/decoding.py#L599)
            text_prompt_ids = text_prompt_ids[-self.config.max_target_positions // 2 - 1 :]
            # Set the decoder_start_token_id to <|startofprev|>
            kwargs.update({"decoder_start_token_id": decoder_start_token_id})

            # If the user passes `max_new_tokens`, increase its number to account for the prompt
            if kwargs.get("max_new_tokens", None) is not None:
                kwargs["max_new_tokens"] += len(text_prompt_ids)
                if kwargs["max_new_tokens"] >= self.config.max_target_positions:
                    raise ValueError(
                        f"The length of the sliced `prompt_ids` is {len(text_prompt_ids)}, and the `max_new_tokens` "
                        f"{kwargs['max_new_tokens'] - len(text_prompt_ids)}. Thus, the combined length of the sliced "
                        f"`prompt_ids` and `max_new_tokens` is: {kwargs['max_new_tokens']}. This exceeds the "
                        f"`max_target_positions` of the Whisper model: {self.config.max_target_positions}. "
                        "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
                        f"so that their combined length is less that {self.config.max_target_positions}."
                    )

            # Reformat the forced_decoder_ids to incorporate the prompt
            non_prompt_forced_decoder_ids = (
                kwargs.pop("forced_decoder_ids", None) or generation_config.forced_decoder_ids
            )
            forced_decoder_ids = [
                *text_prompt_ids,
                generation_config.decoder_start_token_id,
                *[token for _rank, token in non_prompt_forced_decoder_ids],
            ]
            forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_decoder_ids)]
            generation_config.forced_decoder_ids = forced_decoder_ids

        if generation_config.return_timestamps:
            logits_processor = [WhisperTimeStampLogitsProcessor(generation_config)]

        if return_token_timestamps:
            kwargs["output_attentions"] = True
            kwargs["return_dict_in_generate"] = True

            if getattr(generation_config, "task", None) == "translate":
                logger.warning("Token-level timestamps may not be reliable for task 'translate'.")
            if not hasattr(generation_config, "alignment_heads"):
                raise ValueError(
                    "Model generation config has no `alignment_heads`, token-level timestamps not available. "
                    "See https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a on how to add this property to the generation config."
                )

            if kwargs.get("num_frames") is not None:
                generation_config.num_frames = kwargs.pop("num_frames")

        outputs = super().generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            **kwargs,
        )

        if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
            num_frames = getattr(generation_config, "num_frames", None)
            outputs["token_timestamps"] = self._extract_token_timestamps(
                outputs, generation_config.alignment_heads, num_frames=num_frames
            )

        return outputs, generation_config
    