import sys

import re
import numpy as np
import torch

sys.path.append("..")
from models.unimol_tools.utils.util import batch_collate_fn
from models.molbert_tools.data_processor import graph_data_list_to_batch_data

from .constants import (
    MM_MASK_INDEX,
    MM_FILE_TOKENS,
    MM_FILE_I_TOKEN,
    MODALITY_LIST,
)
from .utils import (
    normalize_smiles,
    make_mol_wrap,
)
from .mm_file_loader import load_mm_file


def prepare_mol_figure_2d(modality_inputs, mm_masks):
    # assume the data of figure is path, load data from path
    figure_input = []
    for item in modality_inputs:
        if item['index'] == 0:
            figure_input.append(torch.load(item['data_path'])[0].unsqueeze(0))
        else:
            figure_input.append(torch.load(item['data_path']))

    mol_figure_2d_data = {}
    mol_figure_2d_data['mm_input_type'] = 'raw'
    mol_figure_2d_data['mm_data'] = torch.cat(figure_input, dim=0)
    mol_figure_2d_data['mm_masks'] = mm_masks

    return mol_figure_2d_data

def prepare_mol_figure_2d_data(modality_inputs, mm_masks):
    # assume the data of figure is path, only load path
    figure_input = []
    for item in modality_inputs:
        if item['index'] == 0:
            figure_input.append(item['data_path'][0].unsqueeze(0))
        else:
            figure_input.append(item['data_path'])

    #print(figure_input)
    mol_figure_2d_data = {}
    mol_figure_2d_data['mm_input_type'] = 'raw'
    mol_figure_2d_data['mm_data'] = torch.cat(figure_input, dim=0)
    mol_figure_2d_data['mm_masks'] = mm_masks

    return mol_figure_2d_data

def prepare_mol_geometry_3d(modality_inputs, mm_masks):
    # padding is necessary even for 1 sample (multi mols)
    # -2 is special for mol_geometry, since it has BOS and EOS token in data, but not in embedding
    inner_mm_masks = [torch.ones(len(mol_geometry['src_tokens']) - 2, dtype=torch.long) for mol_geometry in modality_inputs]
    inner_mm_masks = torch.nn.utils.rnn.pad_sequence(inner_mm_masks, batch_first=True, padding_value=0)


    mol_data, _ = batch_collate_fn(modality_inputs)
    mol_geometry_3d_data = {}

    # mol_num_list not used in modeling_chemdfm_3d.py
    #mol_geometry_3d_data['mol_num_list'] = mol_num_list
    mol_geometry_3d_data['mm_input_type'] = 'raw'
    mol_geometry_3d_data['mm_data'] = mol_data
    mol_geometry_3d_data['inner_mm_masks'] = inner_mm_masks
    mol_geometry_3d_data['mm_masks'] = mm_masks # No padding

    return mol_geometry_3d_data

def prepare_mol_graph_2d(modality_inputs, mm_masks):
    # 'batch' tells atom-mol relation, it is not required for single mol
    # we need to add them in batch
    for mol_graph in modality_inputs:
        if 'batch' not in mol_graph.keys():
            mol_graph['batch'] = torch.zeros(mol_graph['x'].shape[0])


    # padding is necessary even for 1 sample (multi mols)
    inner_mm_masks = [torch.ones(len(mol_graph['x']), dtype=torch.long) for mol_graph in modality_inputs]
    inner_mm_masks = torch.nn.utils.rnn.pad_sequence(inner_mm_masks, batch_first=True, padding_value=0)

    mol_data = graph_data_list_to_batch_data(modality_inputs)
    mol_graph_2d_data = {}
    mol_graph_2d_data['mm_input_type'] = 'raw'
    mol_graph_2d_data['mm_data'] = mol_data
    mol_graph_2d_data['inner_mm_masks'] = inner_mm_masks
    mol_graph_2d_data['mm_masks'] = mm_masks # No padding

    return mol_graph_2d_data


def prepare_mol_ms(modality_inputs, mm_masks):
    # padding is necessary even for 1 sample (multi mols)
    peak_list_lst = [peak_list for peak_list in modality_inputs]
    print(peak_list_lst)

    def peak_list_process(peak_list_lst):
        max_mass = 3000.0
        resolution = 0
        mult = pow(10, resolution)
        mz_token_length = int(max_mass * mult)
        pad_token = 0
        mz_token_list_lst = []
        max_token_length = 500
        for peak_list in peak_list_lst:
            peak_list = sorted(peak_list, key=lambda x: x[0])
            mz_token_list = []
            #print(peak_list)
            for mz, _ in peak_list:
                mz_token = int(round(float(mz), resolution) * mult)
                if mz_token > 0 and mz_token < mz_token_length:
                    mz_token_list.append(mz_token)
            if len(mz_token_list) > max_token_length:
                # mz_token_list = mz_token_list[:max_token_length]
                mz_token_list = sorted(random.sample(mz_token_list, max_token_length))
            else:
                mz_token_list.extend([pad_token] * (max_token_length - len(mz_token_list)))
            mz_token_list_lst.append(mz_token_list)
        mz_token_input = torch.tensor(mz_token_list_lst, dtype=torch.long)
        mm_attention_mask = (mz_token_input != pad_token).to(torch.long)
        return mz_token_input, mm_attention_mask

    mz_token_input, mm_attention_mask = peak_list_process(peak_list_lst)

    mol_ms_data = {}
    mol_ms_data['mm_input_type'] = 'raw'
    mol_ms_data['mm_data'] = {"peak_list_input": mz_token_input, "peak_list_mask": mm_attention_mask}
    mol_ms_data['inner_mm_masks'] = mm_attention_mask
    mol_ms_data['mm_masks'] = mm_masks   # No padding

    return mol_ms_data

def prepare_mol_ir(modality_inputs, mm_masks):
    # padding is necessary even for 1 sample (multi mols)
    ir_spectra_lst = [ir_spectra for ir_spectra in modality_inputs]

    def ir_spectra_process(ir_spectra_examples):
        input = {}
        input["ir_spectra_input"] = torch.tensor(np.array(ir_spectra_examples), dtype=torch.float16)
        return input

    ir_input = ir_spectra_process(ir_spectra_lst)

    mol_ir_data = {}
    mol_ir_data['mm_input_type'] = 'raw'
    mol_ir_data['mm_data'] = ir_input
    mol_ir_data['mm_masks'] = mm_masks  # No padding

    return mol_ir_data


def prepare_mm_inputs_masks(final_input,
                            modality_list,
                            input_smiles,
                            mm_data,
                            mm_length,
                            mm_start_id,
                            mm_end_id,
                            mm_pad_id,
                            device,
                            args):
    mm_masks = torch.zeros((5, final_input.size(1)), dtype=torch.long).to(device)
    mm_inputs = {modality:[] for modality in args.modality}

    start_idx = torch.where(final_input == mm_start_id)[1].tolist()
    end_idx = torch.where(final_input == mm_end_id)[1].tolist()

    assert len(start_idx) == len(end_idx)
    is_all_smiles_found = True

    for start, end, modality, single_smiles in zip(start_idx, end_idx, modality_list, input_smiles):
        mm_masks[MM_MASK_INDEX[modality]][start] = 1
        mm_masks[MM_MASK_INDEX[modality]][end] = 2
        mm_masks[MM_MASK_INDEX[modality]][start+1:end] = -1
        final_input[start:end + 1] = mm_pad_id

        # SMILES may be unnormalized in testcase. Our dataset could ensure normalization,
        # but we need to eval on other dataset or benchmarks.
        if single_smiles not in mm_data[modality]:
            single_smiles = normalize_smiles(single_smiles)
        if single_smiles in mm_data[modality]:
            if modality == 'mol_figure_2d':  # Full of Hard coding
                if args.input_file.endswith('reaction_prediction_cllm-cllm.jsonl') or args.input_file.endswith('retro_synthesis_cllm-cllm.jsonl'):
                    mm_inputs[modality].append({'data_path': mm_data[modality][single_smiles], 'index': 0})
                else:
                    mm_inputs[modality].append({'data_path': mm_data[modality][single_smiles], 'index': 1})
                    figure_idx_pos = torch.arange(start, end, 73)[1:] # Hard coding
                    if len(figure_idx_pos) > 0:
                        assert len(figure_idx_pos) > 1
                        if mm_length[modality][single_smiles] > 0:
                            mm_masks[MM_MASK_INDEX[modality]][figure_idx_pos] = torch.arange(len(figure_idx_pos), dtype=torch.long, device='cuda:0') + 3
                        else:
                            mm_masks[MM_MASK_INDEX[modality]][figure_idx_pos[0]] = 3
                            mm_masks[MM_MASK_INDEX[modality]][figure_idx_pos[1:]] = torch.arange(len(figure_idx_pos) - 1, dtype=torch.long, device='cuda:0') + 15
            else:
                mm_inputs[modality].append(mm_data[modality][single_smiles])
        else:
            is_all_smiles_found = False
            print(f'smiles [{single_smiles}] not found in eval, ingore the sample')
            break

        return mm_inputs, mm_masks, is_all_smiles_found


def prepare_infer_data(data):
    # NOTE: in training procedure, we can ensure the SMILES exact match the MM, however, we cannot ensure that in realtime chat.
    # The senerio of realtime chat should be:
    # -> Human: tell me something about the molecule with SMILES: C=COF
    #    Click upload buttom, upload a file.
    # User should not explicitly tell which part is the SMILES, so we don't have the knowledge of SMILES.
    # Therefore, we should not use SMILES as the key index. Instead, we use the pseudo SMILES with index.
    pattern = '|'.join(re.escape(mm_file_token) for mm_file_token in MM_FILE_TOKENS)
    index = 0
    mm_length = {}
    mm_embedding = {}
    for example in data:
        single_mm_token_list = re.findall(pattern, example['input'])
        #print(single_mm_token_list)
        #print(example['mm_input_files'])
        if len(single_mm_token_list) != len(example['mm_input_files']):
            raise ValueError('# of mm tokens in input must match the # of files in mm_input_files')

        example['smiles'] = []
        example['modality'] = []
        for single_mm_token, mm_input_file_name in zip(single_mm_token_list, example['mm_input_files']):
            mm_token_length, mm_inputs = load_mm_file(single_mm_token, mm_input_file_name)

            pseudo_smiles = f'SMILES{index}'
            if single_mm_token == MM_FILE_I_TOKEN:
                mm_length[pseudo_smiles] = mm_token_length
                mm_embedding[pseudo_smiles] = mm_inputs
            else:
                mm_length[pseudo_smiles] = mm_token_length
                mm_embedding[pseudo_smiles] = mm_inputs

            example['smiles'].append(pseudo_smiles)
            example['modality'].append(MODALITY_LIST[MM_FILE_TOKENS.index(single_mm_token)])

            example['input'] = example['input'].replace(single_mm_token, make_mol_wrap(pseudo_smiles), 1)
            #print(mm_length)
            #print(mm_embedding)
            index += 1

    return data, mm_embedding, mm_length

def prepare_interact_data():
    example = {}
    example['instruction'] = input('instruction: ')
    example['input'] = input('input: ')
    example['mm_input_files'] = []

    pattern = '|'.join(re.escape(mm_file_token) for mm_file_token in MM_FILE_TOKENS)
    single_mm_token_list = re.findall(pattern, example['input'])
    mm_token_num = len(single_mm_token_list)
    for i in range(mm_token_num):
        example['mm_input_files'].append(input(f'mm_input_files [{i+1}/{mm_token_num}]: '))

    data = [example]
    return data
