"""
We support direct load mm file from disk, 
"""
import os
import sys
import numpy as np
import torch
import io

from rdkit import Chem
from rdkit.Chem import Draw, AllChem

from PIL import ImageOps

import csv
import json

from transformers import CLIPImageProcessor

from models.unimol_tools.data.conformer import ConformerGen, coords2unimol
from models.molbert_tools.data_processor import mol_to_graph_data_obj_simple
from models.clip_tools.data_processor import process_figure

from .constants import (
    MM_FILE_C_TOKEN,
    MM_FILE_G_TOKEN,
    MM_FILE_I_TOKEN,
    MM_FILE_M_TOKEN,
    MM_FILE_R_TOKEN
)

def load_mol_ms_file(mm_input_file):
    mz_list = []
    intensity_list = []

    # Read the MGF file and process line by line
    if type(mm_input_file) is str:
        with open(mm_input_file, 'rb') as f:
            mm_input_file = f.read()
    f = io.StringIO(mm_input_file.decode('utf-8'))

    spectrum_started = False
    for line in f:
        line = line.strip()

        if line.startswith('BEGIN IONS'):
            spectrum_started = True
            continue
        elif line.startswith('END IONS'):
            spectrum_started = False
            # Process or store data after END IONS if needed
            continue

        if spectrum_started:
            # Split the line into m/z and intensity
            tokens = line.split()
            if len(tokens) == 2:
                mz = float(tokens[0])
                intensity = float(tokens[1])
                intensity_list.append((mz, intensity))


    intensity_list = [intensity_list]
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
            for mz, _ in peak_list:
                mz_token = int(round(float(mz), resolution) * mult)
                if mz_token > 0 and mz_token < mz_token_length:
                    mz_token_list.append(mz_token)
            if len(mz_token_list) > max_token_length:
                mz_token_list = sorted(random.sample(mz_token_list, max_token_length))
            else:
                mz_token_list.extend([pad_token] * (max_token_length - len(mz_token_list)))
            mz_token_list_lst.append(mz_token_list)
        mz_token_input = torch.tensor(mz_token_list_lst, dtype=torch.long)
        mm_attention_mask = (mz_token_input != pad_token).to(torch.long)
        return mz_token_input, mm_attention_mask

    mz_token_input, mm_attention_mask = peak_list_process(intensity_list)
    token_length = torch.sum(mm_attention_mask)

    return token_length, intensity_list[0]

def load_mol_ir_file(mm_input_file):
    ir_data = []

    if type(mm_input_file) is str:
        with open(mm_input_file, mode='rb') as f:
            mm_input_file = f.read()
    f = io.StringIO(mm_input_file.decode('utf-8'))

    csv_reader = csv.reader(f)
    for row in csv_reader:
        ir_data.append(row[1:])

    ir_data = np.array(ir_data, dtype=np.float32)

    return 50, ir_data[1,:]


"""
The length of the figure_2d tokens requires the CLIP model config.
"""
def load_mol_figure_2d_file(mm_input_file):
    processor = CLIPImageProcessor.from_pretrained('./models/clip_tools/clip-vit-L-14-336')

    if type(mm_input_file) is str:
        with open(mm_input_file, 'rb') as f:
            data = f.read()
        mm_input_file = io.BytesIO(data)

    figure, length = process_figure(processor, mm_input_file)
    return length, figure

def load_mol_graph_2d_file(mm_input_file):
    if type(mm_input_file) is str:
        if not mm_input_file.endswith('.sdf'):
            raise ValueError(f'mol_graph_2d now only support .sdf file. {mm_input_file} not supported')

        with open(mm_input_file, 'rb') as f:
            mm_input_file = f.read()

    mol = Chem.MolFromMolBlock(mm_input_file.decode('utf-8'))

    graph = mol_to_graph_data_obj_simple(mol)

    atom_num = len(graph['x'])
    return atom_num, graph

def load_mol_geometry_3d_file(mm_input_file):
    if type(mm_input_file) is str:
        if not mm_input_file.endswith('.xyz'):
            raise ValueError(f'mol_geometry_3d now only support .xyz file. {mm_input_file} not supported')

        with open(mm_input_file, 'rb') as f:
            mm_input_file = f.read()

    mol = Chem.MolFromXYZBlock(mm_input_file.decode('utf-8'))

    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)

    unimol_params = {'data_type':'molecule', 'remove_hs':True}
    conformer = ConformerGen(**unimol_params)

    unimol_data = coords2unimol(atoms, coordinates, conformer.dictionary, conformer.max_atoms, remove_hs=conformer.remove_hs)

    return len(unimol_data['src_tokens']) - 2, unimol_data

def crop(img):
    inv_img = ImageOps.invert(img)
    bbox = inv_img.getbbox()
    bbox = (bbox[0] - 10, bbox[1] - 10, bbox[2] + 10, bbox[3] + 10) 
    img = inv_img.crop(bbox)
    img = ImageOps.invert(img)
    return img

def create_mol_figure_2d_file(smiles, mm_output_file):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(300, 300))
    img = crop(img)
    img.save(mm_output_file)

def create_mol_graph_2d_file(smiles, mm_output_file):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)

    molblock = Chem.MolToMolBlock(mol)
    with open(mm_output_file, 'w') as f:
        f.write(molblock)

"""
Adopted from unimol_tools/data/conformer.py
"""
def create_mol_geometry_3d_file(smiles, mm_output_file):
    seed=42
    mode='fast'
    remove_hs=True

    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert len(atoms)>0, 'No atoms in molecule: {}'.format(smiles)

    res = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if res == 0:
        try:
            # some conformer can not use MMFF optimize
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass
    ## for fast test... ignore this ###
    elif res == -1 and mode == 'heavy':
        AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
        try:
            # some conformer can not use MMFF optimize
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            AllChem.Compute2DCoords(mol)
    else:
        AllChem.Compute2DCoords(mol)

    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smi)

    if remove_hs:
        mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts('[#1]'))

    Chem.MolToXYZFile(mol, mm_output_file)

"""
    Return 2 values:
    1. number of mm tokens, mm_input_data
"""
def load_mm_file(single_mm_token, mm_input_file):
    if single_mm_token == MM_FILE_G_TOKEN:
        return load_mol_graph_2d_file(mm_input_file)
    elif single_mm_token == MM_FILE_C_TOKEN:
        return load_mol_geometry_3d_file(mm_input_file)
    elif single_mm_token == MM_FILE_I_TOKEN:
        return load_mol_figure_2d_file(mm_input_file)
    elif single_mm_token == MM_FILE_M_TOKEN:
        return load_mol_ms_file(mm_input_file)
    elif single_mm_token == MM_FILE_R_TOKEN:
        return load_mol_ir_file(mm_input_file)
    else:
        raise ValueError(f'modality token {single_mm_token} not supported yet.')

def load_json_file(input_file, use_jsonl=True):
    ########################## prepare data #########################
    # pseudo example (input file should be JSONL, but the example is written in JSON):
    #  {
    #   "instruction": "You are a chemist.",
    #   "input": "given the SMILES CCCO and the mol graph <MM_FILE_I> and mol comformation <MM_FILE_C>, tell me if it is BACE.",
    #   "mm_input_files": ["./mol_16888.jpg", "./mol_16888.sdf"]
    #  }
    if not use_jsonl:
        with open(input_file, "r") as reader:
            data = json.load(reader)
    else:
        data = []
        with open(input_file, "r") as reader:
            for line in reader:
                data.append(json.loads(line.strip()))

    return data
