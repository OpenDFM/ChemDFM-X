# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import math
from transformers import (
    AutoConfig,
)
from transformers.deepspeed import HfDeepSpeedConfig

def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    mm_config=None,
                    ds_config=None,
                    rlhf_training=False,
                    dropout=None,
                    torch_dtype=None,
                    cache_dir=None,
                    rope_theta=None,
                    use_flash_attention_2=False,
                    device_map=None):
    model_kwargs = {}
    model_config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    if rope_theta:
        model_config.rope_theta = rope_theta
    model_kwargs['use_flash_attention_2'] = use_flash_attention_2
    configure_dropout(model_config, dropout)

    # add mm_config if needed
    if mm_config is not None:
        modality_hidden_dim = mm_config['mm_hidden_dim']
        model_config.modality = {}
        for m, d in modality_hidden_dim.items():
            model_config.modality[m] = {'mm_projector_type': 'mlp2x_gelu', 'mm_hidden_size': d}
            if m == 'mol_figure_2d':
                from transformers import CLIPVisionConfig
                model_config.figure_config = CLIPVisionConfig.from_pretrained('./models/clip_tools/clip-vit-L-14-336')
                model_config.modality[m]['conv_shape'] = '1x8'
    else:
        from transformers import CLIPVisionConfig
        model_config.figure_config = CLIPVisionConfig.from_pretrained('./models/clip_tools/clip-vit-L-14-336')

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
        # If you have extra parameters do not use this. It may init some of the parameters on meta device which may cause undesired performance
        model_kwargs['low_cpu_mem_usage'] = False
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            cache_dir=cache_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **model_kwargs)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model
