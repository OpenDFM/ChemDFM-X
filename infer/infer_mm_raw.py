# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# It takes inputs and raw mm file as input. (image/mol/ms/ir)
#
# NOTE: We support several input keys, like
#   1. "prompt", "question"
#   2. "instruction", "input"
#   3. "question"
# Here we choose "instruction"+"input", if change, need to check prompts.py
# Since the modality has long and different length, it is not proper to ask user to
# provide the modality token length, so we need to parse it.
# So, the input keys should be:
#   1. "instruction"
#   2. "input": contains mol_tokens, that each mol correpound to only 1 token related to the modality
#                  <MM_FILE_G>/<MM_FILE_I> and etc. This token is first converted to multiple mm_tokens before passing to the tokenizer.
#   3. "mm_input_files": a list of mm files coorespound to the mol_tokens.

import os
import sys
import json
import torch
import logging
import argparse
from tqdm import tqdm

sys.path.append("..")
from dschat.utils.utils import load_hf_tokenizer
from dschat.utils.model.model_utils_figure import create_hf_model

from models.modeling_chemdfm_defaultmm_lora import ChemDFMMMForPreTraining

from .constants import (
    MOLSTART,
    MOLEND,
    ERRORSTART,
    ERROREND,
    DEFAULT_PAD_TOKEN,
    DEFAULT_MM_START,
    DEFAULT_MM_END,
    DEFAULT_MM_TOKENS,
    MM_MASK_INDEX,
    MM_FILE_TOKENS,
    MM_FILE_I_TOKEN,
    MODALITY_LIST
)

from .prompts import prompt_input_normal as prompt_input
from .utils import (
    generate,
    normalize_smiles,
    to_device,
    choice_answer_postprocess,
    text_answer_postprocess,
    make_mol_wrap
)

from .prepare_data import (
    prepare_mol_figure_2d,
    prepare_mol_figure_2d_data,
    prepare_mol_geometry_3d,
    prepare_mol_graph_2d,
    prepare_mol_ms,
    prepare_mol_ir,
    prepare_infer_data,
    prepare_interact_data
)

from .mm_file_loader import load_json_file

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help='Specify num of top k',
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help='Specify num of top p',
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help='Specify num of repetition penalty',
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help='Specify temperature',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help="input file path",
        default=""
    )

    parser.add_argument(
        '--mm_input_format',
        default='only_mm',
        type=str,
        choices=['only_mm', 'only_text', 'both']
    )

    parser.add_argument(
        '--output_file',
        type=str,
        help='output file path',
        default=""
    )

    parser.add_argument(
        '--mm_file_cache',
        default=None,
        type=str,
        help='the mm files are stored as a binary streams in .pt file. to reduce too many small files.'
    )

    parser.add_argument(
        '--use_jsonl',
        action='store_true',
        help='if use json instead of jsonl, the input json file should already be a list.'
    )

    parser.add_argument(
        '--text_answer',
        action='store_true'
    )

    parser.add_argument(
        '--without_wrapping',
        action='store_true'
    )

    parser.add_argument(
        '--device_map',
        default=None
    )

    parser.add_argument(
        '--conversational',
        action='store_true'
    )

    parser.add_argument(
        '--interact',
        action='store_true'
    )


    parser.add_argument(
        '--no_few_shots',
        action='store_true'
    )

    args = parser.parse_args()

    return args



def response(args,
             model,
             tokenizer,
             mm_length,
             mm_embedding,
             inputs,
             device,
             modality_list):
    '''
    main procedure to inference with modality inputs.
    '''
    print("################## new response #################")
    print(inputs)
    print(modality_list)
    print("#################################################")
    if MOLSTART not in inputs or MOLEND not in inputs:
        if MOLSTART in inputs or MOLEND in inputs:
            print(f'WARNING: {MOLSTART} and {MOLEND} token not matched.')
        else:
            print('Processing with NO mm inputs.')
        final_input = tokenizer(inputs, return_tensors="pt").to(device).input_ids
        mm_masks = None
        mm_inputs = None
    else:
        mm_pad_id = tokenizer.convert_tokens_to_ids([DEFAULT_PAD_TOKEN])[0]
        mm_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_MM_TOKENS])[0]
        mm_start_id = tokenizer.convert_tokens_to_ids([DEFAULT_MM_START])[0]
        mm_end_id = tokenizer.convert_tokens_to_ids([DEFAULT_MM_END])[0]

        print('Processing with mm inputs')
        print('Raw inputs:')
        print(inputs)
        #############  1. parse the wrapped MM_FILE_TOKENS into MM_TOKENS(of desired length) #############
        inputs_split = inputs.split(MOLSTART)
        input_text, input_smiles = inputs_split[0], []

        # NOTE: The (+ '1') procedure is to fix the special token issue(unexpected space) in tokenizer
        # However, this might be fixed in later transformer versions (maybe already fixed, but we didn't update)
        # So, if we change it for newer transformer, we need to change other corresponding code in this file.
        try:
            smiles_index = 0
            for split in inputs_split[1:]:
                split_smiles, split_text = split.split(MOLEND)
                processed_smiles = ''

                # NOTE: The "if" structure cannot change!!! The reaction_picture_recognition task takes the whole reaction as key.
                # Therefore, the procedure must not enter the split analyze!
                if 'reaction_picture_recognition' in args.input_file:
                    processed_smiles += DEFAULT_MM_START + DEFAULT_MM_TOKENS * (abs(mm_length[split_smiles]) * 73 - 1) + DEFAULT_MM_END + '1'
                    input_smiles.append(split_smiles)
                else:
                    for smiles_by_dot in split_smiles.split('.'):
                        for smiles in smiles_by_dot.split('>'):
                            if smiles != '':
                                modality = modality_list[smiles_index]
                                smiles_index += 1
                                if modality == 'mol_figure_2d':
                                    processed_smiles += DEFAULT_MM_START + DEFAULT_MM_TOKENS * (mm_length[smiles] * 73 - 1) + DEFAULT_MM_END + '1'
                                elif modality == 'mol_ir' or modality == 'mol_ms' or modality == 'mol_geometry_3d' or modality == 'mol_graph_2d':
                                    processed_smiles += DEFAULT_MM_START + DEFAULT_MM_TOKENS * mm_length[smiles] + DEFAULT_MM_END + '1'
                                else:
                                    raise ValueError(f'modality: {modality} not supported.')
                                input_smiles.append(smiles)

                            processed_smiles += '>'
                        processed_smiles = processed_smiles[:-1] + '.'
                processed_smiles = processed_smiles[:-1]
                input_text += processed_smiles + split_text
        except KeyError as e:
            print(f'Smiles {e} not found in input \'{inputs}\'')
            return 'Smiles not found'
        print('\nProcessed inputs:')
        print(input_text)


        ###################### 2. prepare the tokens and masks #########################
        ### NOTE: step 2 of fix the unexpected space issue
        final_input = tokenizer(input_text).input_ids
        end_index = 0  # FIXME hard code for ignore the space
        while mm_end_id in final_input[end_index:]:
            tmp = final_input[end_index:].index(mm_end_id)
            end_index += tmp + 1
            final_input = final_input[:end_index] + final_input[end_index + 2:]
        final_input = torch.tensor([final_input], dtype=torch.long).to(device)

        input_ids = final_input[0]
        start_idx = torch.where(input_ids == mm_start_id)[0].tolist()
        end_idx = torch.where(input_ids == mm_end_id)[0].tolist()

        mm_masks = torch.zeros((len(MODALITY_LIST), input_ids.size(0)), dtype=torch.long).to(device)
        for start, end, mod in zip(start_idx, end_idx, modality_list):
            mod_idx = MODALITY_LIST.index(mod)
            mm_masks[mod_idx][start] = 1
            mm_masks[mod_idx][end] = 2
            mm_masks[mod_idx][start + 1:end] = -1
            input_ids[start:end + 1] = mm_pad_id

        ##################### new mm_inputs for raw input ###########################
        # the batch is also build here with size 1,
        mm_data = mm_embedding
        mm_inputs = {modality:[] for modality in MODALITY_LIST}

        ##################### Special treatment for figure ##########################
        # NOTE: In normal inference, multiple modality might be provided, so, though start_idx
        # and end_idx are only used for figure, we first compute them in all cases.
        start_idx = torch.where(mm_masks[2] == 1)[0].tolist()  # Hard coding [0]
        end_idx = torch.where(mm_masks[2] == 2)[0].tolist()
        assert len(start_idx) == len(end_idx)

        is_all_smiles_found = True
        for index, single_smiles in enumerate(input_smiles):
            # SMILES may be unnormalized in testcase. Our dataset could ensure normalization,
            # but we need to eval on other dataset or benchmarks.
            if single_smiles not in mm_data:
                single_smiles = normalize_smiles(single_smiles)

            if single_smiles in mm_data:
                modality = modality_list[index]
                if modality == 'mol_figure_2d':  # FIXME: Hard coding
                    mm_inputs[modality].append({'data_path': mm_data[single_smiles], 'index': 1})
                    figure_idx_pos = torch.arange(start_idx.pop(0), end_idx.pop(0), 73)[1:] # Hard coding
                    if len(figure_idx_pos) > 0:
                        assert len(figure_idx_pos) > 1
                        if mm_length[single_smiles] > 0:
                            mm_masks[2][figure_idx_pos] = torch.arange(len(figure_idx_pos), dtype=torch.long, device='cuda:0') + 3
                        else:
                            mm_masks[2][figure_idx_pos[0]] = 3
                            mm_masks[2][figure_idx_pos[1:]] = torch.arange(len(figure_idx_pos) - 1, dtype=torch.long, device='cuda:0') + 15
                else:
                    mm_inputs[modality].append(mm_data[single_smiles])
            else:
                is_all_smiles_found = False
                print(f'smiles [{single_smiles}] not found in eval, ingore the sample')
                break

        #TODO: decide how to deal with the not found smiles
        if not is_all_smiles_found:
            raise ValueError('smiles not found, not handled')

        ############### prepare modality data in encoder input structure ############
        for modality in MODALITY_LIST:
            if len(mm_inputs[modality]) == 0:
                mm_inputs[modality] == None
                continue

            if modality == 'mol_figure_2d':
                mm_inputs['mol_figure_2d'] = prepare_mol_figure_2d_data(mm_inputs[modality], mm_masks)
            elif modality == 'mol_geometry_3d':
                mm_inputs['mol_geometry_3d'] = prepare_mol_geometry_3d(mm_inputs[modality], mm_masks)
            elif modality == 'mol_graph_2d':
                mm_inputs['mol_graph_2d'] = prepare_mol_graph_2d(mm_inputs[modality], mm_masks)
            elif modality == 'mol_ms':
                mm_inputs['mol_ms'] = prepare_mol_ms(mm_inputs[modality], mm_masks)
            elif modality == 'mol_ir':
                mm_inputs['mol_ir'] = prepare_mol_ir(mm_inputs[modality], mm_masks)
            else:
                raise ValueError(f'modality {modality} not supported')
        mm_inputs = to_device(mm_inputs, device)
        #############################################################################

        # Add unsqueeze to support multi-modality in 1 prompt
        mm_masks = mm_masks.unsqueeze(0)  # (1, N) -> (B, N)

        print(f'\nOrigin input: {inputs}')

    try:
        gen_output = generate(model, tokenizer, final_input, mm_masks, mm_inputs, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens, do_sample=False)
    except torch.cuda.OutOfMemoryError:
        return "Input too long"

    post_func = choice_answer_postprocess if not args.text_answer else text_answer_postprocess
    print(gen_output[0])
    print()
    return post_func(gen_output[0], cut_prefix=len(tokenizer.decode(final_input[0], skip_special_tokens=True)))


def replace_filename_with_filestream(data, mm_file_cache):
    """
        The modality files could be stored by filename or filestreams.
        If already stored in filestreams in cache, direct load it here.
    """
    if mm_file_cache is not None:
        file_cache = torch.load(mm_file_cache)
    else:
        file_cache = {}

    for item in data:
        mm_input_files = item['mm_input_files']
        replace_files = []
        for mm_file in mm_input_files:
            if mm_file in file_cache:
                replace_files.append(file_cache[mm_file])
            else:
                replace_files.append(mm_file)
        item['mm_input_files'] = replace_files

    return data

def main():
    args = parse_args()

    ########################## prepare model ########################
    print("Loading tokenizer...")
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)

    print("Loading model...")
    model = create_hf_model(ChemDFMMMForPreTraining, args.model_name_or_path, tokenizer, device_map=args.device_map).half()
    if args.device_map is None:
        device = torch.device("cuda:0")
        model = model.to(device)
    model.eval()
    #################################################################

    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    with open(args.output_file, "w") as writer:
        while True:
            if args.interact:
                data = prepare_interact_data()
            else:
                data = load_json_file(args.input_file, args.use_jsonl)

            data = replace_filename_with_filestream(data, args.mm_file_cache)

            data, mm_embedding, mm_length = prepare_infer_data(data)

            for example in tqdm(data):
                example["pred"] = response(
                    args,
                    model,
                    tokenizer,
                    mm_length,
                    mm_embedding,
                    prompt_input(example, args, shorter=10000, conversational=args.conversational),
                    device,
                    example['modality']
                )
                example["mm_input_files"] = []

                writer.write(json.dumps(example) + '\n')
                writer.flush()

            if not args.interact:
                break

if __name__ == "__main__":
    main()
