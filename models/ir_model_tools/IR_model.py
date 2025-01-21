import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig


class IRSpectraCollator(object):
    def __init__(self, **kwargs):
        self.tokenizer = kwargs.pop("tokenizer") if "tokenizer" in kwargs.keys() else None
        self.max_length = kwargs.pop("max_length") if "max_length" in kwargs.keys() else 512
        self.remain_raw_data = kwargs.pop("remain_raw_data") if "remain_raw_data" in kwargs.keys() else False

    def __call__(self, ir_spectra_examples):
        input = {}
        ir = []
        for ir_spectra_example in ir_spectra_examples:
            ir.append(np.array(ir_spectra_example)[:, 1])
        input["ir_spectra_input"] = torch.tensor(ir, dtype=torch.float32)
        return input
    
    def train_call(self, examples):
        input = {}
        smi = []
        ir = []
        for example in examples:
            smi.append(example["smiles"])
            ir.append(np.array(example["ir_spectra"])[:, 1])
        output = self.tokenizer(smi, padding=True, max_length=self.max_length, truncation=True, return_tensors="pt")
        input["labels"] = output["input_ids"][:, 1:]
        input["decoder_input_ids"] = output["input_ids"][:, :-1]
        input["ir_spectra_input"] = torch.tensor(ir, dtype=torch.float32)
        if self.remain_raw_data: input["raw_data"] = {"smiles": smi}
        return input


class Linear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "relu":
            self._trunc_normal_init(2.0)
        elif init == "glorot":
            self._glorot_uniform_init()
        elif init == "gating":
            self._zero_init(self.use_bias)
        elif init == "normal":
            self._normal_init()
        elif init == "final":
            self._zero_init(False)
        elif init == "jax":
            self._jax_init()
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale**0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        # nn.init.trunc_normal_(self.weight, mean=0.0, std=std)
        tmp = self.weight.cuda()
        nn.init.trunc_normal_(tmp, mean=0.0, std=std)
        self.weight = nn.Parameter(tmp.cpu())

    def _glorot_uniform_init(self):
        nn.init.xavier_uniform_(self.weight, gain=1)

    def _zero_init(self, use_bias=True):
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)

    def _normal_init(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

    def _jax_init(self):
        input_size = self.weight.shape[-1]
        std = math.sqrt(1 / input_size)
        nn.init.trunc_normal_(self.weight, std=std, a=-2.0 * std, b=2.0 * std)


class MLP(nn.Module):
    def __init__(
        self,
        d_in,
        n_layers,
        d_hidden,
        d_out,
        activation=nn.ReLU(),
        bias=True,
        final_init="final",
    ):
        super(MLP, self).__init__()
        layers = [Linear(d_in, d_hidden, bias), activation]
        for _ in range(n_layers):
            layers += [Linear(d_hidden, d_hidden, bias), activation]
        layers.append(Linear(d_hidden, d_out, bias, init=final_init))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class MolecularGenerator(nn.Module):
    def __init__(self, config_json_path, tokenizer_path):
        super().__init__()
        with open(config_json_path, "r") as f:
            self.model = BartForConditionalGeneration(config=BartConfig(**json.loads(f.read())))
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
        self.mlp = MLP(36, 3, 512, 768, activation=nn.ReLU())

    def ir_projection(self, ir_spectra_input):
        #ir_spectra_input = ir_spectra_input.reshape([ir_spectra_input.size(0), 50, 36])
        ir_spectra_input = ir_spectra_input.contiguous().view(ir_spectra_input.size(0), 50, 36)
        ir_embedding = self.mlp(ir_spectra_input)
        return ir_embedding

    def forward(self, **kwargs):
        ir_embedding = self.ir_projection(kwargs.pop("ir_spectra_input"))
        return self.model(inputs_embeds=ir_embedding, **kwargs)

    def infer(self, num_beams=10, num_return_sequences=None, max_length=512, **kwargs):
        ir_embedding = self.ir_projection(kwargs.pop("ir_spectra_input"))
        res_lst = self.model.generate(
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_beams
            if num_return_sequences is None
            else num_return_sequences,
            inputs_embeds=ir_embedding,
            decoder_start_token_id=0,
        )
        smiles_lst = [self.tokenizer.decode(out).replace("<pad>", "").replace("<s>", "").replace("</s>", "") for out in res_lst]
        batch_size = ir_embedding.size(0)
        num_return_sequences = len(smiles_lst) // batch_size
        infer_smiles_lst = []
        for i in range(0, batch_size * num_return_sequences, num_return_sequences):
            infer_smiles_lst.append(smiles_lst[i:i+num_return_sequences])
        return infer_smiles_lst
    
    def encoding(self, **kwargs):
        ir_input = kwargs.pop("ir_spectra_input")
        # self.mlp.to(dtype=ir_input.dtype)
        # self.model.to(dtype=ir_input.dtype)
        ir_embedding = self.ir_projection(ir_input)
        encoder = self.model.get_encoder()
        encoder_outputs = encoder(inputs_embeds=ir_embedding, **kwargs)
        return encoder_outputs

    def load_model(self, path):
        if path is not None:
            model_dict = torch.load(path, map_location=torch.device("cpu"))
            self.load_state_dict(model_dict)


def to_device(batch, device):
        output = {}
        for k, v in batch.items():
            try:
                output[k] = v.to(device)
            except:
                output[k] = v
        return output


# input: [ir_spectra_1, ir_spectra_2, ..., ir_spectra_n]
# output: torch.tensor[n, 50, 768]
def ir_spectra_encoding(ir_spectra_list, device="cuda"):
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "IR_final_model.pt")
    model = MolecularGenerator(
        config_json_path=os.path.join(current_dir, "config_dir/configs/bart.json"),
        tokenizer_path=os.path.join(current_dir, "config_dir/tokenizer-smiles-bart/"),
    )
    model.load_model(model_path)
    model.to(device)
    model.eval()
    ir_spectra_input = IRSpectraCollator()(ir_spectra_list)
    ir_spectra_input = to_device(ir_spectra_input, device)
    ir_spectra_embedding = model.encoding(**ir_spectra_input).last_hidden_state
    return ir_spectra_embedding


if __name__=="__main__":
    pass
