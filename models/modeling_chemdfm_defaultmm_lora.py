from typing import List, Optional, Tuple, Union

import os
import re
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import LlamaForCausalLM, LlamaPreTrainedModel, CLIPVisionModel, PreTrainedModel, PretrainedConfig
from einops import rearrange
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.unimol_tools.models.unimol import UniMolModel
from models.molbert_tools.molbert import GNN_graphpred as MolBERT
from models.ms_model_tools.MS2_model import MolecularGenerator as MSMolecularGenerator
from models.ir_model_tools.IR_model import MolecularGenerator as IRMolecularGenerator


class MplugDocOwlHReducerModel(PreTrainedModel):
    def __init__(self, config, config_dict):
        super().__init__(config)
        self.config = config
        self.config_dict = config_dict
        self.ln_q = torch.nn.LayerNorm(self.config_dict['mm_hidden_size'], eps=1e-6)
        self.conv_shape = (int(self.config_dict['conv_shape'].split('x')[0]), int(self.config_dict['conv_shape'].split('x')[1])) # 
        self.conv_patch=self.conv_shape[0]*self.conv_shape[1]
        ## feature interaction with a conv layer
        self.reducer_before = torch.nn.Sequential(
            nn.Conv2d(self.config_dict['mm_hidden_size'], self.conv_patch*self.config_dict['mm_hidden_size'], kernel_size=self.conv_shape, stride=self.conv_shape, bias=True),
            nn.GELU()
        )
        ## reduce visual feature length with a conv layer
        self.reducer = nn.Conv2d(self.config_dict['mm_hidden_size'], self.config_dict['mm_hidden_size'], kernel_size=self.conv_shape, stride=self.conv_shape, bias=True)    
        ## align visual features with language embedding with fc
        self.visual_fc = torch.nn.Linear(self.config_dict['mm_hidden_size'], self.config_dict['hidden_size'])

        self.post_init()

    def forward(
        self,
        encoder_hidden_states=None
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            batch_size is the number of all images (global+crop) in a batch
            Sequence of hidden-states at the output of the last layer of the encoder.
        """
        encoder_hidden_states = encoder_hidden_states[:,1:,:] # remove the first cls token 
        B, L, C = encoder_hidden_states.shape # B, 1024=(448/14)^2, 1024

        ## feature interaction with a conv layer
        encoder_hidden_states = rearrange(encoder_hidden_states, 'B (H W) D -> B D H W', H=int(math.sqrt(L)))
        hidden_states = self.reducer_before(encoder_hidden_states) # B 4D H W/4
        ## reduce seq length with a conv layer
        hidden_states = rearrange(hidden_states, 'B (X D) H W -> B D H (W X)', X=self.conv_patch) # B 4D H W/4 -> B D H W
        sequence_output = self.reducer(hidden_states) # B,C,H,W -> B,C,H/conv_shape[0],W/(conv_shape[1])
        sequence_output = sequence_output.flatten(2).transpose(1, 2)  # B,C,H/conv_shape[0],W/(conv_shape[1]) -> B,C,L/conv_patch -> B,L/conv_patch,C
        sequence_output = sequence_output.transpose(0, 1).contiguous() # L/conv_patch, B, C
        ## align visual features with language embedding with fc
        sequence_output = self.visual_fc(sequence_output) # L/conv_patch, B, h
        sequence_output = sequence_output.transpose(0, 1).contiguous() # B, s/4, h

        return sequence_output


# -1: content; 0: padding, 1: start, 2:end
class FigureProjector(LlamaPreTrainedModel):
    def __init__(self, config, config_dict, hidden_size):
        super(FigureProjector, self).__init__(config)
        self.config = config
        self.config_dict = config_dict
        self.config_dict['hidden_size'] = hidden_size
        self.h_reducer = MplugDocOwlHReducerModel(config.figure_config, self.config_dict)
        self.projector_type = config_dict['mm_projector_type'] if 'mm_projector_type' in config_dict else 'linear'
        self.marker = nn.Embedding(26, self.config_dict['hidden_size'])  # Hard coding: only consider 13 figures (including full figure)
        # 1(pad) + 2 (start and end) + 12 (x = 1, y = 1 ... 12) + 11 (x = 2 ... 12, y = 1)
        self.post_init()

    def forward(self, inputs, mm_mask):
        outputs = self.h_reducer(inputs)
        outputs = torch.flatten(outputs, 0, 1)
        mm_mask[mm_mask == -1] = 0
        marker = self.marker(mm_mask)
        marker[mm_mask == 0] = 0
        return outputs, marker

    def get_projector_layer(self):
        if self.projector_type == 'linear':
            return nn.Linear(self.config_dict['mm_hidden_size'], self.config_dict['hidden_size'])

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(self.config_dict['mm_hidden_size'], self.config_dict['hidden_size'])]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(self.config_dict['hidden_size'], self.config_dict['hidden_size']))
            return nn.Sequential(*modules)

        if self.projector_type == 'identity':
            return IdentityMap()

        if self.projector_type == 'qformer':
            raise NotImplementedError()

        raise ValueError(f'Unknown projector type: {self.projector_type}')


# -1: content; 0: padding, 1: start, 2:end
class IRProjector(LlamaPreTrainedModel):
    def __init__(self, config, config_dict, hidden_size):
        super(IRProjector, self).__init__(config)

        self.config_dict = config_dict
        self.config_dict['hidden_size'] = hidden_size
        self.projector_type = config_dict['mm_projector_type'] if 'mm_projector_type' in config_dict else 'linear'

        self.marker = nn.Embedding(3, self.config_dict['hidden_size'])
        self.projector = self.get_projector_layer()
        self.post_init()

    def forward(self, inputs, mm_mask):
        outputs = self.projector(inputs)
        outputs = torch.flatten(outputs, 0, 1)
        mm_mask[mm_mask == -1] = 0
        marker = self.marker(mm_mask)
        marker[mm_mask == 0] = 0
        return outputs, marker

    def get_projector_layer(self):
        if self.projector_type == 'linear':
            return nn.Linear(self.config_dict['mm_hidden_size'], self.config_dict['hidden_size'])

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(self.config_dict['mm_hidden_size'], self.config_dict['hidden_size'])]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(self.config_dict['hidden_size'], self.config_dict['hidden_size']))
            return nn.Sequential(*modules)

        if self.projector_type == 'identity':
            return IdentityMap()

        if self.projector_type == 'qformer':
            raise NotImplementedError()

        raise ValueError(f'Unknown projector type: {self.projector_type}')


# -1: content; 0: padding, 1: start, 2:end
class MMOnlyProjector(LlamaPreTrainedModel):
    def __init__(self, config, config_dict, hidden_size):
        super(MMOnlyProjector, self).__init__(config)

        self.config_dict = config_dict
        self.config_dict['hidden_size'] = hidden_size
        self.projector_type = config_dict['mm_projector_type'] if 'mm_projector_type' in config_dict else 'linear'

        self.marker = nn.Embedding(3, self.config_dict['hidden_size'])
        self.projector = self.get_projector_layer()
        self.post_init()

    def forward(self, inputs, mm_mask, inner_mm_masks):
        outputs = self.projector(inputs)
        outputs = outputs[torch.where(inner_mm_masks == 1)]
        mm_mask[mm_mask == -1] = 0
        marker = self.marker(mm_mask)
        marker[mm_mask == 0] = 0
        return outputs, marker

    def get_projector_layer(self):
        if self.projector_type == 'linear':
            return nn.Linear(self.config_dict['mm_hidden_size'], self.config_dict['hidden_size'])

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(self.config_dict['mm_hidden_size'], self.config_dict['hidden_size'])]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(self.config_dict['hidden_size'], self.config_dict['hidden_size']))
            return nn.Sequential(*modules)

        if self.projector_type == 'identity':
            return IdentityMap()

        if self.projector_type == 'qformer':
            raise NotImplementedError()

        raise ValueError(f'Unknown projector type: {self.projector_type}')

def create_modality_projector(modality, config_dict, config):
    if modality == 'mol_geometry_3d':
        return MMOnlyProjector(config, config_dict, config.hidden_size)
    elif modality == 'mol_graph_2d':
        return MMOnlyProjector(config, config_dict, config.hidden_size)
    elif modality == 'mol_figure_2d':
        return FigureProjector(config, config_dict, config.hidden_size)
    elif modality == 'mol_ms':
        return MMOnlyProjector(config, config_dict, config.hidden_size)
    elif modality == 'mol_ir':
        return IRProjector(config, config_dict, config.hidden_size)
    else:
        raise ValueError(f'modality {modality} not supported')

class ChemDFMMMForPreTraining(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.modality = config.modality
        self.projector = nn.ModuleDict({m: create_modality_projector(m, c, config) for m, c in self.modality.items()})


        self.unimol_encoder = UniMolModel(output_dim=1, data_type='molecule', remove_hs=True)
        self.mol_graph_2d_encoder = MolBERT(num_layer=5, emb_dim=300, num_tasks=2, JK='last', drop_ratio=0.5, graph_pooling='mean', gnn_type='gin')  # Hard coding
        self.mol_figure_encoder = CLIPVisionModel(config.figure_config)

        # FIXME hard code
        current_dir = "./models/ir_model_tools"
        self.mol_ir_model = IRMolecularGenerator(
            config_json_path=os.path.join(current_dir, "config_dir/configs/bart.json"),
            tokenizer_path=os.path.join(current_dir, "config_dir/tokenizer-smiles-bart/"),
        )
        self.mol_ir_encoder = self.mol_ir_model.encoding

        # FIXME hard code
        current_dir = "./models/ms_model_tools"
        self.mol_ms_model = MSMolecularGenerator(
            config_json_path=os.path.join(current_dir, "config_dir/configs/bart.json"),
            tokenizer_path=os.path.join(current_dir, "config_dir/tokenizer-smiles-bart/"),
        )
        self.mol_ms_encoder = self.mol_ms_model.encoding

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        mm_masks: Optional[torch.LongTensor] = None,  # size: (batch_size, num_modality, sequence_length)
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **mm_inputs: Optional[dict[str, torch.Tensor]]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print(input_ids)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        if mm_masks is not None:
            projector = None
            # NOTE: Create dummy inputs is necessary in training
            for idx, (modality, projector) in enumerate(self.projector.items()):
                if modality not in mm_inputs or mm_inputs[modality] is None or (mm_masks[:, idx] != 0).sum() == 0:
                    if modality == 'mol_geometry_3d':
                        tmp_input = {}
                        tmp_input['src_tokens'] = torch.tensor([[1,4,2]], dtype=torch.long, device=inputs_embeds.device)
                        tmp_input['src_coord'] = torch.ones((1, 3, 3), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                        tmp_input['src_distance'] = torch.ones((1, 3, 3), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                        tmp_input['src_edge_type'] = torch.zeros((1, 3, 3), dtype=torch.long, device=inputs_embeds.device) + 128
                        tmp_inner_mm_masks = torch.ones((1, 1), dtype=torch.long, device=inputs_embeds.device)

                        modality_data = {}
                        modality_data['mm_data'] = tmp_input
                        modality_data['inner_mm_masks'] = tmp_inner_mm_masks
                    elif modality == 'mol_graph_2d':
                        tmp_input = torch.tensor(((5, 0),), dtype=torch.long, device=inputs_embeds.device)
                        tmp_edge_index = torch.empty((2, 0), dtype=torch.long, device=inputs_embeds.device)
                        tmp_edge_attr = torch.empty((0, 2), dtype=torch.long, device=inputs_embeds.device)
                        tmp_batch = torch.zeros(tmp_input.shape[0], dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                        tmp_inner_mm_masks = torch.ones((1, 1), dtype=torch.long, device=inputs_embeds.device)
                        mm_data = {}
                        mm_data['x'] = tmp_input
                        mm_data['edge_index'] = tmp_edge_index
                        mm_data['edge_attr'] = tmp_edge_attr
                        mm_data['batch'] = tmp_batch
                        modality_data = {}
                        modality_data['mm_data'] = mm_data
                        modality_data['inner_mm_masks'] = tmp_inner_mm_masks
                    elif modality == 'mol_figure_2d':
                        tmp_input = torch.zeros((1, 3, 336, 336), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                        modality_data = {}
                        modality_data['mm_data'] = tmp_input
                    elif modality == 'mol_ms':
                        tmp_input = torch.zeros((1, 500), dtype=torch.int64, device=inputs_embeds.device)
                        tmp_add = torch.ones((1, 500), dtype=torch.int64, device=inputs_embeds.device)

                        mm_data = {}
                        mm_data['peak_list_input'] = tmp_input
                        mm_data['peak_list_mask'] = tmp_add
                        modality_data = {}
                        modality_data['mm_data'] = mm_data
                        modality_data['inner_mm_masks'] = tmp_add
                    elif modality == 'mol_ir':
                        tmp_input = torch.zeros((1, 1800), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                        mm_data = {}
                        mm_data['ir_spectra_input'] = tmp_input
                        modality_data = {}
                        modality_data['mm_data'] = mm_data

                    modality_data['mm_input_type'] = 'raw'
                    default_weight = 1.0 # use original value, not add computed values
                    example_weight = 0.0
                else:
                    modality_data = mm_inputs[modality]
                    default_weight = 0.0
                    example_weight = 1.0 # use computed values

                if modality_data['mm_input_type'] == 'raw':
                    mm_data = modality_data['mm_data']

                    if modality == 'mol_geometry_3d':
                        y = self.unimol_encoder(**mm_data, return_repr=True, return_atomic_reprs=True)
                        atom_repr = y['atomic_reprs']
                        pad_y = pad_sequence(atom_repr, padding_value=0, batch_first=True)

                        inner_mm_masks = modality_data['inner_mm_masks']
                        mm_proj, marker = projector(pad_y, mm_masks[:, idx].clone(), inner_mm_masks)
                    elif modality == 'mol_graph_2d':
                        y = self.mol_graph_2d_encoder(mm_data['x'], mm_data['edge_index'], mm_data['edge_attr'], mm_data['batch'])
                        pad_y = pad_sequence(y, padding_value=0, batch_first=True)

                        inner_mm_masks = modality_data['inner_mm_masks']
                        mm_proj, marker = projector(pad_y, mm_masks[:, idx].clone(), inner_mm_masks)
                    elif modality == 'mol_figure_2d':
                        y = self.mol_figure_encoder(mm_data).last_hidden_state
                        pad_y = y

                        mm_proj, marker = projector(pad_y, mm_masks[:, idx].clone())
                    elif modality == 'mol_ms':
                        y = self.mol_ms_encoder(**mm_data).last_hidden_state
                        inner_mm_masks = modality_data['inner_mm_masks']
                        pad_y = y

                        mm_proj, marker = projector(pad_y, mm_masks[:, idx].clone(), inner_mm_masks)
                    elif modality == 'mol_ir':
                        y = self.mol_ir_encoder(**mm_data).last_hidden_state
                        pad_y = y

                        mm_proj, marker = projector(pad_y, mm_masks[:, idx].clone())
                    else:
                        raise ValueError(f'modality not supported')

                    ## inputs_embeds: B * T * D, mm_masks: B * X * T
                    # TODO: when training with unseen modality in real data, the expected value of
                    # 'torch.where(mm_masks[:, idx] == 16888)' and 'torch.where(mm_masks[:, idx] != 0)' are same ([]).
                    # But for debugging, set mm_masks requires in-place operations, which might cause extra bugs.
                    # So, we make the separate operations instead of 'inputs_embeds[torch.where(mm_masks[:, idx] != 0)] = 0'.
                    inputs_embeds = inputs_embeds.clone()
                    if default_weight == 1:
                        inputs_embeds[torch.where(mm_masks[:, idx] == 16888)] = 0 # torch.where shoule be empty, to enable all_gather across nodes
                        inputs_embeds[torch.where(mm_masks[:, idx] == 16888)] = inputs_embeds[torch.where(mm_masks[:, idx] == 16888)] * default_weight + mm_proj[0:0] * example_weight
                    else:
                        inputs_embeds[torch.where(mm_masks[:, idx] != 0)] = 0
                        inputs_embeds[torch.where(mm_masks[:, idx] == -1)] = inputs_embeds[torch.where(mm_masks[:, idx] == -1)] * default_weight  + mm_proj[:] * example_weight

                    # a+=b and a=a+b are DIFFERENT! a+=b is in-place operation!
                    inputs_embeds = inputs_embeds + marker * example_weight # example: 1, default: 0
                else:
                    raise ValueError(f'mm_input_type not supported')

        inputs_embeds = inputs_embeds.clone()

        returns = super().forward(
            None, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict
        )
        return returns

    def prepare_inputs_for_generation(
        self, input_ids, mm_masks=None, mm_inputs=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        else:
            remove_prefix_length = 0

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if mm_masks is not None:
            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                    "mm_masks": mm_masks[:, :, remove_prefix_length:] if remove_prefix_length < mm_masks.shape[2] else None,
                }
            )
        else:
            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                }
            )
        if mm_inputs is not None:
            model_inputs.update(mm_inputs)
        return model_inputs

    def save_projectors(self, output_dir):
        for modality, projector in self.projector.items():
            torch.save(projector, os.path.join(output_dir, f'{modality}.bin'))
