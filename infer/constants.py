# Tokens in training/eval jsonl
MOLSTART = '[ChemDFM_Start_SMILES]'
MOLEND = '[ChemDFM_End_SMILES]'
ERRORSTART = '[Chem_Start_SMILES]'
ERROREND = '[Chem_End_SMILES]'

# Tokens for tokenizer
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_MM_START = '<|mm_start|>'
DEFAULT_MM_END = '<|mm_end|>'
DEFAULT_MM_TOKENS = '<|mm_tokens|>'

# mm_mask index in model inference
MM_MASK_INDEX = {
    'mol_geometry_3d': 0,
    'mol_graph_2d': 1,
    'mol_figure_2d':2,
    'mol_ms':3,
    'mol_ir':4
}
MODALITY_LIST = [
    'mol_geometry_3d',
    'mol_graph_2d',
    'mol_figure_2d',
    'mol_ms',
    'mol_ir'
]

# Tokens in new inference by user
MM_FILE_C_TOKEN = '[MM_FILE_C]'
MM_FILE_G_TOKEN = '[MM_FILE_G]'
MM_FILE_I_TOKEN = '[MM_FILE_I]'
MM_FILE_M_TOKEN = '[MM_FILE_M]'
MM_FILE_R_TOKEN = '[MM_FILE_R]'
MM_FILE_TOKENS = [
    MM_FILE_C_TOKEN,
    MM_FILE_G_TOKEN,
    MM_FILE_I_TOKEN,
    MM_FILE_M_TOKEN,
    MM_FILE_R_TOKEN
]

# Input replacement token in raw text
TEXT_REPLACE_TOKEN = {
    'mol_geometry_3d': 'mol_3d',
    'mol_graph_2d': 'mol_2d',
    'mol_figure_2d': 'mol_figure',
    'mol_ms': 'spectra_ms',
    'mol_ir': 'spectra_ir'
}

