import torch
import torch.nn as nn
from torch import Tensor, device
from sklearn.preprocessing import normalize
import os
from transformers import AutoTokenizer, AutoModel
# from peft import LoraConfig, get_peft_model
from typing import List, Optional

class EncodingModel_Stella(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        model_dir = self.config.model_dir

        vector_dim = 768
        vector_linear_directory = f"2_Dense_{vector_dim}"
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.vector_linear = torch.nn.Linear(in_features=self.model.config.hidden_size, out_features=vector_dim)
        vector_linear_dict = {
            k.replace("linear.", ""): v for k, v in
            torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
        }
        self.vector_linear.load_state_dict(vector_linear_dict)
        self.vector_linear.cuda()


        # self.model = self.initialize_peft(self.model)

    def initialize_peft(self, model, lora_r: int = 512, lora_alpha: int = 1024, lora_dropout: float = 0.05):
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        return model

    def forward(self, inputs):
        batch_size = len(inputs)
#         self.model.train()
        input_data = self.tokenizer(inputs, padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_data = {k: v.cuda() for k, v in input_data.items()}
        attention_mask = input_data["attention_mask"]
        last_hidden_state = self.model(**input_data)[0]
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        
        # mask_positions = input_data["input_ids"] == self.tokenizer.mask_token_id
        # mask_embeddings = last_hidden[mask_positions]
        return vectors
        # vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    def forward_mixup(self, inputs):
        batch_size = len(inputs)
#         self.model.train()
        input_data = self.tokenizer(inputs, padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_data = {k: v.cuda() for k, v in input_data.items()}
        attention_mask = input_data["attention_mask"]
        last_hidden_state = self.model(**input_data)[0]
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        
        masks1 = []
        masks2 = []
        for i in range(batch_size):
            current_hidden = last_hidden[i][input_data['attention_mask'][i] == 1]
            # mask_positions = input_data["input_ids"][i] == self.tokenizer.mask_token_id
            mask_position = torch.where(input_data['input_ids'][i] == self.tokenizer.mask_token_id)[0]
            mask_position = mask_position[0].item()
            mask1 = current_hidden[:mask_position]
            mask2 = current_hidden[mask_position+1:]
            vector1 = mask1.sum(dim=0) / mask1.shape[0]
            vector2 = mask2.sum(dim=0) / mask2.shape[0]
            masks1.append(vector1)
            masks2.append(vector2)

            # res.append()
            # mask_embeddings = last_hidden[i][mask_positions] # 2 x 1024
            # if mask_embeddings.shape[0] != 2:
            #     print('mask_embeddings.shape[0] != 2')
            #     print(mask_embeddings.shape)
            #     print(inputs[i])
            # res.append(mask_embeddings)

        # res = torch.stack(res) # batch_size x 2 x 1024
        # mask1 = res[:, 0, :]
        # mask2 = res[:, 1, :]
        mask1 = torch.stack(masks1)
        mask2 = torch.stack(masks2)
        return mask1, mask2

        # mask_positions = input_data["input_ids"] == self.tokenizer.mask_token_id
        # mask_hidden_states = last_hidden[mask_positions]
        # mask_hidden_states = mask_hidden_states.view(batch_size, 2, -1)
        # masks_1 = mask_hidden_states[:, 0, :]
        # masks_2 = mask_hidden_states[:, 1, :]
        # return masks_1, masks_2
        
    
    def compress_to_bert_size(self, vectors):
        return self.vector_linear(vectors)
        
        