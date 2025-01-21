from rdkit import Chem
from .constants import MOLSTART, MOLEND

'''
GenerationConfig(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=100
)
'''
def generate(model,
             tokenizer,
             inputs,
             mm_masks,
             mm_inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=True,
             num_return_sequences=1,
             max_new_tokens=100):

    if mm_masks is None:
        generate_ids = model.generate(inputs,
                                      num_beam_groups=num_beam_groups,
                                      num_beams=num_beams,
                                      do_sample=do_sample,
                                      num_return_sequences=num_return_sequences,
                                      max_new_tokens=max_new_tokens,
                                      #   repetition_penalty=1.2,
                                      temperature=0.2,
                                      top_k=40,
                                      top_p=0.9,
                                      eos_token_id = tokenizer.eos_token_id)
    else:
        generate_ids = model.generate(inputs,
                                      mm_masks=mm_masks,
                                      mm_inputs=mm_inputs,
                                      num_beams=num_beams,
                                      num_beam_groups=num_beam_groups,
                                      do_sample=do_sample,
                                      num_return_sequences=num_return_sequences,
                                      max_new_tokens=max_new_tokens,
                                      #   repetition_penalty=1.2,
                                      temperature=0.2,
                                      top_k=40,
                                      top_p=0.9,
                                      eos_token_id = tokenizer.eos_token_id) ## generation ends with eos_token


    try:
        result = tokenizer.batch_decode(generate_ids,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
    except IndexError:
        result = 'Error Decoding'
    return result

def make_mol_wrap(mol_str):
    return MOLSTART + mol_str + MOLEND

# the input smiles may not be normalized
def normalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


# recursively to_device
def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        if type(v) == dict:
            output[k] = to_device(v, device)
        else:
            try:
                output[k] = v.to(device)
            except:
                output[k] = v
    return output

def choice_answer_postprocess(answer, cut_prefix=0):
    try:
        return answer[cut_prefix:].split()[0].strip()
    except:
        return answer[cut_prefix:].strip()

def text_answer_postprocess(answer, cut_prefix):
    return answer[cut_prefix:].strip()

