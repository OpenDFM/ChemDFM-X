#!/bin/bash

MAX_NEW_TOKENS=1024

MODEL_NAME_OR_PATH=./checkpoints/ChemDFM-X-v1.0-13B

python -u -m infer.infer_mm_raw \
       --model_name_or_path ${MODEL_NAME_OR_PATH} \
       --max_new_tokens $MAX_NEW_TOKENS \
       --output_file ./output.jsonl \
       --use_jsonl \
       --text_answer \
       --conversational \
       --interact

