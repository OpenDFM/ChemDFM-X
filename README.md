# ChemDFM-X: Towards Large Multimodal Model for Chemistry


## Index
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Example](#example)
- [TODO]

## Introduction
[ChemDFM-X](https://www.sciengine.com/SCIS/doi/10.1007/s11432-024-4243-0) is a multimodal model for chemisty, supporting 5 modality files: molecule graph (2D), molecule comformer (3D), molecule picture, mass spectra (MS) and infrared spectrum (IR).
Every modality data is encoded by a modality encoder.

<img src="./ChemDFM-X.png" alt="ChemDFM-X introduction." width="600">

## Getting Started
1. Download ChemDFM-X model parameters from [HuggingFace](https://huggingface.co/OpenDFM/ChemDFM-X-v1.0-13B) or [ModelScope](https://modelscope.cn/models/OpenDFM/ChemDFM-X-v1.0-13B).
2. Install the required libs. The prefered enviroment is listed in requirements.txt. We strongly suggest installing PyTorch, PyTorch-Geometry and Uni-Mol first before the other requirements.
`pip install -r requirements.txt`

## Usage
1. Run the bash command to launch the demo. Please ensure your environment is activated.
 * bash ./infer/scripts/interact.sh
2. Give instruction.
3. Give input text mixed with modality tokens (1 token for each file).
4. Give real file path to each of the modality token one by one.

The specital tokens for each modality is listed:


 | modality | modality token | file format | 
 |  :--- | :--- |
 | molecule Graph | [MM_FILE_G] | mol.sdf |
 | molecule Comformer | [MM_FILE_C] | mol.xyz |
 | molecule Image | [MM_FILE_I] | mol.png |
 | Mass spectra | [MM_FILE_M] | mol.csv |
 | infRaraed spectrum | [MM_FILE_R] | mol.mgf |

 ## Example
More examples will be updated later.

 | instruction | input | mm_input_files |
 |  :--- | :--- | :--- |
 | Would you please predict the SMILES notation that corresponds to the molecular figure? | **[MM_FILE_I]** | ./example/C=COF.png |
 | | | |
 | As a seasoned chemist, you have the SMILES notation with molecular graph of the identified reactants, reagents and products from an incomplete chemical reaction. It appears that some component or components in the products are missing. Using the information presented in the remaining parts of the reaction equation, could you make an educated guess about what these missing substances could be? Please confine your answer to the SMILES of the unknown molecule(s) and avoid incorporating any superfluous information. | SMILES of Reactants: CC(C)[Mg]Cl.CSc1c(F)cc(F)cc1Br.COB(OC)OC \n molecular graph of Reactants **[MM_FILE_G] [MM_FILE_G] [MM_FILE_G]**\nSMILES of Reagents: C1CCOC1\nmolecular graph of Reagents: **[MM_FILE_G]**\nSMILES of Products:\nmolecular graph of Products:\nSMILES of the absent products:\nAssistant:|CC(C)[Mg]Cl.sdf CSc1c(F)cc(F)cc1Br.sdf COB(OC)OC.sdf C1CCOC1.sdf

