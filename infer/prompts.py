from .constants import *

def prompt_input_normal(raw, args, shorter=0, conversational=False, token_keys=["mm"]):
    def construct_input(input_format, human):
        print(input_format)
        print(human)
        if args.mm_input_format == 'both':
            if isinstance(human, tuple):
                try:
                    return input_format.format(
                        smiles=human[0].replace(MOLSTART, '').replace(MOLEND, '').strip(),
                        spectra=human[1].strip()
                    )
                except KeyError:
                    token_replace = {key:human[1].strip() for key in token_keys}
                    return input_format.format(
                        smiles=human[0].replace(MOLSTART, '').replace(MOLEND, '').strip(),
                        #mm=human[1].strip()
                        **token_replace
                    )
            else:
                token_replace = {key:human for key in token_keys}
                return input_format.format(
                    smiles=human.replace(MOLSTART, '').replace(MOLEND, '').strip(),
                    **token_replace
                    #mm=human.strip()
                )
        else:
            try:
                #return input_format.format(mm=human.strip())
                token_replace = {key:human.strip() for key in token_keys}
                return input_format.format(**token_replace)
            except IndexError:
                return input_format.format(human.strip())

    wrapping = not args.without_wrapping
    if 'few_shots_examples' in raw:
        if args.input_file.endswith('spectrum.jsonl'):
            if isinstance(raw['answer'], list):
                tmp_answer = f'{MOLSTART}{".".join(raw["answer"])}{MOLEND}'
            else:
                tmp_answer = f'{MOLSTART}{raw["answer"]}{MOLEND}'
        else:
            tmp_answer = None
        if conversational:
            # version 1
            # FIXME: Hard coding
            if raw['question'] != '':
                if raw['question'].endswith(' Several examples will be provided.\n'):
                    text = f"[Round 0]\nHuman: {raw['question'][:-36]}\nAssistant: OK."
                else:
                    text = f"[Round 0]\nHuman: {raw['question'].strip()}\nAssistant: OK."
                for idx, fs in enumerate(raw['few_shots_examples'][shorter:]):
                    # if args.mm_input_format == 'both':
                    #     pass
                    # else:
                    cur_input = construct_input(
                        raw['input_format'], fs['human'] if not args.input_file.endswith('spectrum.jsonl')
                        else (fs['human'], f'{MOLSTART}{fs["assistant"]}{MOLEND}')
                        )
                    text += f"\n[Round {idx + 1}]\nHuman: {cur_input}\nAssistant: {fs['assistant'].strip()}"
                final_input = construct_input(
                    raw['input_format'], raw['input'].strip() if not args.input_file.endswith('spectrum.jsonl')
                    else (raw['input'].strip(), tmp_answer)
                    )
                text += f"\n[Round {len(raw['few_shots_examples']) + 1}]\nHuman: {final_input}\nAssistant:"
            else:
                text = ''
                for idx, fs in enumerate(raw['few_shots_examples'][shorter:]):
                    cur_input = construct_input(
                        #TODO: Some scripts has '.strip()', some not, maybe deleted later.
                        #raw['input_format'], fs['human'].strip())
                        raw['input_format'], fs['human'].strip() if not args.input_file.endswith('spectrum.jsonl') else (fs['human'], f'{MOLSTART}{fs["assistant"]}{MOLEND}')
                        )
                    text += f"[Round {idx}]\nHuman: {cur_input}\nAssistant: {fs['assistant'].strip()}\n"
                final_input = construct_input(
                    raw['input_format'], raw['input'].strip() if not args.input_file.endswith('spectrum.jsonl') else (raw['input'].strip(), tmp_answer)
                    )
                text += f"[Round {len(raw['few_shots_examples'])}]\nHuman: {final_input}\nAssistant:"
        else:
            text = raw['question']
            for fs in raw['few_shots_examples'][shorter:]:
                text += construct_input(
                        raw['input_format'], fs['human'] if not args.input_file.endswith('spectrum.jsonl') else (fs['human'], f'{MOLSTART}{fs["assistant"]}{MOLEND}')
                        ) + f' {fs["assistant"]}\n'
            text += construct_input(
                    raw['input_format'], raw['input'].strip() if not args.input_file.endswith('spectrum.jsonl') else (raw['input'].strip(), tmp_answer)
                    )
            if wrapping:
                text = f"[Round 0]\nHuman: {text}\nAssistant:"
            # elif not args.galactica:
            #     text = f"[INST] {text.strip()} [/INST]"
    elif 'prompt' in raw:
        text = raw['prompt'] + '\n\n' + raw['question']
        if wrapping:
            text = f"[Round 0]\nHuman: {text}\nAssistant:"
        # elif not args.galactica:
        #     text = f"[INST] {text.strip()} [/INST]"
    elif 'instruction' in raw:
        assert 'input' in raw and 'question' not in raw
        text = raw['instruction'].strip() + '\n' + raw['input'].strip()
        if wrapping:
            text = f"[Round 0]\nHuman: {text}\nAssistant:"
    else:
        text = raw['question']
        if wrapping:
            text = f"[Round 0]\nHuman: {text}\nAssistant:"
        # elif not args.galactica:
        #     text = f"[INST] {text.strip()} [/INST]"
    # FIXME: hard coding for correction
    return text.replace('molecule formular', 'molecular formula').strip()


def prompt_input_rocauc(raw, args, shorter=0, conversational=False, token_keys=["mm"]):
    def construct_input(input_format, human):
        print(input_format)
        print(human)
        token_replace = {key:human.strip() for key in token_keys}
        if args.mm_input_format == 'both':
            return input_format.format(
                smiles=human.replace(MOLSTART, '').replace(MOLEND, '').strip(),
                **token_replace
                #mm=human.strip()
            )
        else:
            try:
                #return input_format.format(mm=human.strip())
                return input_format.format(**token_replace)
            except IndexError:
                return input_format.format(human.strip())

    wrapping = not args.without_wrapping
    if 'few_shots_examples' in raw:
        if conversational:
            # version 1
            # text = f"[Round 0]\nHuman: {raw['question'].strip()}\nAssistant: OK."
            # for idx, fs in enumerate(raw['few_shots_examples'][shorter:]):
            #     text += f"\n[Round {idx + 1}]\nHuman: {construct_input(raw['input_format'], fs['human'].strip())}\nAssistant: {fs['assistant'].strip()}"
            # text += f"\n[Round {len(raw['few_shots_examples']) + 1}]\nHuman: {construct_input(raw['input_format'], raw['input'].strip())}\nAssistant:"
            # print(text)
            # version 2
            # text = f"[Round 0]\nHuman: {raw['question'].strip()}\n"
            # candidate = raw['few_shots_examples'][shorter:]
            # if len(candidate) > 0:
            #     text += f"{construct_input(raw['input_format'], candidate[0]['human'].strip())}\nAssistant: {candidate[0]['assistant'].strip()}"
            #     for idx, fs in enumerate(candidate[1:]):
            #         text += f"\n[Round {idx + 1}]\nHuman: {construct_input(raw['input_format'], fs['human'].strip())}\nAssistant: {fs['assistant'].strip()}"
            #     text += f"\n[Round {len(candidate)}]\nHuman: {construct_input(raw['input_format'], raw['input'].strip())}\nAssistant:"
            # else:
            #     text += f"{construct_input(raw['input_format'], raw['input'].strip())}\nAssistant:"
            # version 3
            text = ''
            for idx, fs in enumerate(raw['few_shots_examples'][shorter:]):
                text += f"[Round {idx}]\nHuman: {raw['question'].strip()}\n{construct_input(raw['input_format'], fs['human'].strip())}\nAssistant: {fs['assistant'].strip()}\n"
            text += f"[Round {len(raw['few_shots_examples'])}]\nHuman: {raw['question'].strip()}\n{construct_input(raw['input_format'], raw['input'].strip())}\nAssistant:"
        else:
            text = raw['question']
            for fs in raw['few_shots_examples'][shorter:]:
                text += construct_input(raw['input_format'], fs['human']) + f' {fs["assistant"]}\n'
            text += construct_input(raw['input_format'], raw['input'])
            if wrapping:
                text = f"[Round 0]\nHuman: {text}\nAssistant:"
            # else:
            #     text = f"[INST] {text.strip()} [/INST]"
    else:
        text = raw['question']
        if wrapping:
            text = f"[Round 0]\nHuman: {text}\nAssistant:"
        # else:
        #     text = f"[INST] {text.strip()} [/INST]"
    return text

