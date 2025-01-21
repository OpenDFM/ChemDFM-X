import os
import random
import json
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig


class MS2SpectraCollator(object):
    def __init__(self, **kwargs):
        self.tokenizer = kwargs.pop("tokenizer") if "tokenizer" in kwargs.keys() else None
        self.max_length = kwargs.pop("max_length") if "max_length" in kwargs.keys() else 512
        self.remain_raw_data = kwargs.pop("remain_raw_data") if "remain_raw_data" in kwargs.keys() else True

    def __call__(self, ms2_spectra_examples):
        input = {}
        input["peak_list_input"], input["peak_list_mask"] = self.peak_list_process(ms2_spectra_examples)
        return input

    def train_call(self, example_lst):
        input = {}
        peak_list_lst = []
        smiles_lst = []
        for example in example_lst:
            peak_list_lst.append(example["peak_list"])
            smiles_lst.append(example["smiles"])
        if self.remain_raw_data:
            input["raw_data"] = {}
            input["raw_data"]["peak_list"] = peak_list_lst
            input["raw_data"]["smiles"] = smiles_lst
        input["peak_list_input"], input["peak_list_mask"] = self.peak_list_process(peak_list_lst)
        output = self.tokenizer(smiles_lst, padding=True, max_length=self.max_length, truncation=True, return_tensors="pt")
        input["labels"] = output["input_ids"][:, 1:].contiguous()
        input["labels"][input["labels"] == self.tokenizer.pad_token_id] = -100
        input["decoder_input_ids"] = output["input_ids"][:, :-1].contiguous()
        return input

    def peak_list_process(self, peak_list_lst):
        max_mass = 3000.0
        resolution = 0
        mult = pow(10, resolution)
        mz_token_length = int(max_mass * mult)
        pad_token = 0
        mz_token_list_lst = []
        max_token_length = 500      # 注意最大长度不能超过模型最大position embedding的长度512
        for peak_list in peak_list_lst:
            peak_list = sorted(peak_list, key=lambda x: x[0])
            mz_token_list = []
            for mz, _ in peak_list:
                mz_token = int(round(float(mz), resolution) * mult)
                if mz_token > 0 and mz_token < mz_token_length:
                    mz_token_list.append(mz_token)
            if len(mz_token_list) > max_token_length:
                # mz_token_list = mz_token_list[:max_token_length]
                mz_token_list = sorted(random.sample(mz_token_list, max_token_length))
            else: mz_token_list.extend([pad_token] * (max_token_length - len(mz_token_list)))
            mz_token_list_lst.append(mz_token_list)
        mz_token_input = torch.tensor(mz_token_list_lst, dtype=torch.long)
        attention_mask = (mz_token_input != pad_token).to(torch.long)
        return mz_token_input, attention_mask


class MolecularGenerator(nn.Module):
    def __init__(self, config_json_path, tokenizer_path):
        super().__init__()
        with open(config_json_path, "r") as f:
            self.model = BartForConditionalGeneration(config=BartConfig(**json.loads(f.read())))
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
        self.mz_token_embedding = nn.Embedding(3000, 512)      # vocab size与collate func中的mz_token_length对齐

    def ms2_projection(self, peak_list_input, peak_list_mask):
        ms2_embedding = self.mz_token_embedding(peak_list_input)
        ms2_mask = peak_list_mask
        return ms2_embedding, ms2_mask

    def forward(self, **kwargs):
        ms2_embedding, ms2_mask = self.ms2_projection(kwargs.pop("peak_list_input"), kwargs.pop("peak_list_mask"))
        return self.model(inputs_embeds=ms2_embedding, attention_mask=ms2_mask, **kwargs)

    def infer(self, num_beams=10, num_return_sequences=None, max_length=512, **kwargs):
        ms2_embedding, ms2_mask = self.ms2_projection(kwargs.pop("peak_list_input"), kwargs.pop("peak_list_mask"))
        res_lst = self.model.generate(
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_beams
            if num_return_sequences is None
            else num_return_sequences,
            length_penalty=0.0,
            inputs_embeds=ms2_embedding,
            attention_mask=ms2_mask,
            decoder_start_token_id=0,
        )
        smiles_lst = [self.tokenizer.decode(out).replace("<pad>", "").replace("<s>", "").replace("</s>", "") for out in res_lst]
        batch_size = ms2_embedding.size(0)
        num_return_sequences = len(smiles_lst) // batch_size
        infer_smiles_lst = []
        for i in range(0, batch_size * num_return_sequences, num_return_sequences):
            infer_smiles_lst.append(smiles_lst[i:i+num_return_sequences])
        return infer_smiles_lst
    
    def encoding(self, **kwargs):
        peak_list_input = kwargs.pop("peak_list_input")
        attention_mask = kwargs.pop("peak_list_mask")
        ms2_embedding, ms2_mask = self.ms2_projection(peak_list_input, attention_mask)
        # encoder = self.model.get_encoder()
        encoder_outputs = self.model.get_encoder()(inputs_embeds=ms2_embedding, attention_mask=ms2_mask, **kwargs)
        return encoder_outputs

    def load_model(self, path):
        if path is not None:
            model_dict = torch.load(path, map_location=torch.device("cpu"))
            self.load_state_dict(model_dict)
            # print(self.model.model.encoder.gradient_checkpointing)


def to_device(batch, device):
        output = {}
        for k, v in batch.items():
            try:
                output[k] = v.to(device)
            except:
                output[k] = v
        return output


# input: [ms2_spectra_1, ms2_spectra_2, ..., ms2_spectra_n]
# output: torch.tensor[n, 512, 768]
def ms2_spectra_encoding(ms2_spectra_list):
    current_dir = os.path.dirname(__file__)
    device = "cuda"
    model_path = os.path.join(current_dir, "MS2_final_model.pt")
    model = MolecularGenerator(
        config_json_path=os.path.join(current_dir, "config_dir/configs/bart.json"),
        tokenizer_path=os.path.join(current_dir, "config_dir/tokenizer-smiles-bart/"),
    )
    model.load_model(model_path)
    model.to(device)
    model.eval()
    ms2_spectra_input = MS2SpectraCollator()(ms2_spectra_list)
    ms2_spectra_input = to_device(ms2_spectra_input, device)
    ms2_spectra_embedding = model.encoding(**ms2_spectra_input).last_hidden_state
    return ms2_spectra_embedding


##################################################以上均为模型相关##################################################


# import math
# import time
# from torch.utils.data import DataLoader
# from transformers import AdamW, SchedulerType, get_scheduler, set_seed
# from dataset import TrainMS2Dataset


# def train(model_path=None,
#           per_device_train_batch_size=4,
#           learning_rate=5e-5,
#           weight_decay=0,
#           num_train_epochs=None,
#           max_train_steps=20,
#           gradient_accumulation_steps=1,
#           lr_scheduler_type="linear",
#           num_warmup_epochs=0,
#           output_dir="train_models",
#           seed=42,
#           device="cpu"
#         ):
#     if output_dir is not None:
#         os.makedirs(output_dir, exist_ok=True)
        
#     if seed is not None:
#         set_seed(seed)

#     model = MolecularGenerator(
#         config_json_path="config_dir/configs/bart.json",
#         tokenizer_path="config_dir/tokenizer-smiles-bart/",
#     )
#     model.load_model(model_path)
#     model = model.to(device)

#     train_dataloader = DataLoader(
#         TrainMS2Dataset(),
#         shuffle=True,
#         collate_fn=MS2SpectraCollator(tokenizer=model.tokenizer, remain_raw_data=False).train_call,
#         batch_size=per_device_train_batch_size,
#         num_workers=16,
#     )

#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [
#                 p
#                 for n, p in model.named_parameters()
#                 if not any(nd in n for nd in no_decay)
#             ],
#             "weight_decay": weight_decay,
#         },
#         {
#             "params": [
#                 p
#                 for n, p in model.named_parameters()
#                 if any(nd in n for nd in no_decay)
#             ],
#             "weight_decay": 0.0,
#         },
#     ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, no_deprecation_warning=True)

#     model.train()

#     num_update_steps_per_epoch = math.ceil(
#         len(train_dataloader) / gradient_accumulation_steps
#     )
#     if max_train_steps is None:
#         max_train_steps = num_train_epochs * num_update_steps_per_epoch
#     else:
#         num_train_epochs = math.ceil(
#             max_train_steps / num_update_steps_per_epoch
#         )

#     lr_scheduler = get_scheduler(
#         name=lr_scheduler_type,
#         optimizer=optimizer,
#         num_warmup_steps=num_warmup_epochs * num_update_steps_per_epoch,
#         num_training_steps=max_train_steps,
#     )
#     start_epoch = 0
#     completed_steps = 0

#     print(f"start train! total epoch {num_train_epochs}.")
#     print("original learning rate: {}".format(optimizer.state_dict()["param_groups"][0]["lr"]))
#     for epoch in range(start_epoch, num_train_epochs):
#         train_loss_sum = 0.0
#         start_time = time.time()
#         print(f"current epoch {epoch + 1}:")
#         print("start pred data train!")
#         for step, batch in enumerate(train_dataloader):
#             batch = to_device(batch, device)
#             outputs = model(**batch)
#             loss = outputs.loss
#             loss = loss / gradient_accumulation_steps
#             loss.backward()
#             train_loss_sum += loss.item()
#             if (
#                 step % gradient_accumulation_steps == 0
#                 or step == len(train_dataloader) - 1
#             ):
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()
#                 completed_steps += 1

#             print_step = 1
#             if (step + 1) % print_step == 0:
#                 print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(
#                     epoch + 1,
#                     step + 1,
#                     len(train_dataloader),
#                     train_loss_sum / (step + 1),
#                     time.time() - start_time,
#                 ))
#                 # print("Learning rate = {}".format(
#                 #     optimizer.state_dict()["param_groups"][0]["lr"]
#                 # ))
#             if completed_steps >= max_train_steps:
#                 break

#         if completed_steps >= max_train_steps:
#             break
#         save_model_name = f"epoch{epoch+1}_model.pt"
#         print(f"save the model ({save_model_name}) ...")
#         torch.save(model.state_dict(), os.path.join(output_dir, save_model_name))
#     save_model_name = f"final_model.pt"
#     print(f"save the final model ({save_model_name}) ...")
#     torch.save(model.state_dict(), os.path.join(output_dir, save_model_name))


# def main():
#     train(model_path=None,
#           per_device_train_batch_size=32,
#           learning_rate=5e-5,
#           weight_decay=1e-2,
#           num_train_epochs=5,
#           max_train_steps=None,
#           gradient_accumulation_steps=1,
#           lr_scheduler_type="linear",
#           num_warmup_epochs=0,
#           output_dir="train_models",
#           seed=42,
#           device="cuda"
#         )


# if __name__ == "__main__":
#     main()
