import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor, device
from sklearn.preprocessing import normalize
import os
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, PeftModel
from typing import List, Optional
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint



class EncodingModel_Stella(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        model_dir = self.config.model_dir

        vector_dim = 768
        vector_linear_directory = f"2_Dense_{vector_dim}"
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda()
        
        for param in self.model.parameters():
            param.requires_grad = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        # self.vector_linear = torch.nn.Linear(in_features=self.model.config.hidden_size, out_features=vector_dim)
        self.vector_linear = nn.Sequential(
            nn.Linear(in_features=self.model.config.hidden_size, out_features=vector_dim),
            nn.ReLU(),
            nn.Linear(in_features=vector_dim, out_features=vector_dim),
            nn.Tanh()
        )
        # vector_linear_dict = {
        #     k.replace("linear.", ""): v for k, v in
        #     torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
        # }
        # self.vector_linear.load_state_dict(vector_linear_dict)
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
        # last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        last_hidden = last_hidden_state
        mask_positions = input_data["input_ids"] == self.tokenizer.mask_token_id
        mask_embeddings = last_hidden[mask_positions]
        
        assert mask_embeddings.shape[0] == batch_size, f"{mask_embeddings.shape[0]} != {batch_size}, {inputs[0]}"
        assert mask_embeddings.shape[1] == 1024, f"{mask_embeddings.shape[1]} != 1024"
        return mask_embeddings

        # vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        
        # mask_positions = input_data["input_ids"] == self.tokenizer.mask_token_id
        # mask_embeddings = last_hidden[mask_positions]
        # return vectors
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
            mask_position = torch.where(input_data['input_ids'][i] == self.tokenizer.sep_token_id)[0]
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
        
        




class EncodingModel_Jina(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        model_dir = self.config.model_dir

        vector_dim = 768
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.vector_linear = nn.Sequential(
            nn.Linear(in_features=self.model.config.hidden_size, out_features=vector_dim),
            nn.ReLU(),
            nn.Linear(in_features=vector_dim, out_features=vector_dim),
            nn.Tanh()
        )
        self.vector_linear.cuda()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, inputs):
        batch_size = len(inputs)
#         self.model.train()
        input_data = self.tokenizer(inputs, padding="longest", truncation=True, max_length=512, return_tensors="pt").to(self.model.device)
        task = 'classification'
        task_id = self.model._adaptation_map[task]
        adapter_mask = torch.full((len(inputs),), task_id, dtype=torch.int32).to(self.model.device)
        model_output = self.model(**input_data, adapter_mask=adapter_mask)
        embeddings = self.mean_pooling(model_output, input_data["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

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
            mask_position = torch.where(input_data['input_ids'][i] == self.tokenizer.sep_token_id)[0]
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



class EncodingModel_mxbai(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        model_dir = self.config.model_dir

        vector_dim = 768
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.vector_linear = nn.Sequential(
            nn.Linear(in_features=self.model.config.hidden_size, out_features=vector_dim),
            nn.ReLU(),
            nn.Linear(in_features=vector_dim, out_features=vector_dim),
            nn.Tanh()
        )
        self.vector_linear.cuda()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, inputs):
        batch_size = len(inputs)
#         self.model.train()
        input_data = self.tokenizer(inputs, padding="longest", truncation=True, max_length=512, return_tensors="pt").to(self.model.device)
        task = 'classification'
        task_id = self.model._adaptation_map[task]
        adapter_mask = torch.full((len(inputs),), task_id, dtype=torch.int32).to(self.model.device)
        model_output = self.model(**input_data, adapter_mask=adapter_mask)
        embeddings = self.mean_pooling(model_output, input_data["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    def compress_to_bert_size(self, vectors):
        return self.vector_linear(vectors)





class EncodingModel_NVembed(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        model_dir = self.config.model_dir

        vector_dim = 768
        self.model = model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True).cuda()
        lora_config = LoraConfig(
            r=16,               # Rank of the low-rank matrices
            lora_alpha=32,      # Scaling factor
            lora_dropout=0.1,   # Dropout rate
            bias="none",        # Whether or not to include bias terms in the LoRA layers
            target_modules=["q_proj", "v_proj"]  # Specific layers to apply LoRA to
        )
        self.model = get_peft_model(model, lora_config)

        self.tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
        self.vector_linear = nn.Sequential(
            nn.Linear(in_features=self.model.config.hidden_size, out_features=vector_dim),
            nn.ReLU(),
            nn.Linear(in_features=vector_dim, out_features=vector_dim),
            nn.Tanh()
        )
        
        # self.scaler = GradScaler()

        # self.vector_linear.cuda()

    def forward(self, inputs):
        input_data = self.tokenizer(inputs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_data = {k: v.to(self.model.device) for k, v in input_data.items()}
        
        # Unpack input_data
        input_ids = input_data['input_ids']
        attention_mask = input_data['attention_mask']
        
        # Use gradient checkpointing
        outputs = checkpoint(self.model, input_ids, attention_mask)
        embeddings = outputs['sentence_embeddings'].mean(dim=1)
        return embeddings

    def compress_to_bert_size(self, vectors):
        return self.vector_linear(vectors)

    def to(self, device):
        self.model = self.model.to(device)
        self.vector_linear = self.vector_linear.to(device)
        return self

from llm2vec import LLM2Vec
class EncodingModel_LLM2vec(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
                "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            )
        self.encoder = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            merge_peft=True,
            pooling_mode="mean",
            max_length=256,
            skip_instruction = False,
            
        )
        self.encoder.model = self.initialize_peft(
            self.encoder.model,
        )
            
    def initialize_peft(
        self,
        model,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_modules: Optional[List[str]] = None,
    ):
        if lora_modules is None and model.config.__class__.__name__ in [
            "LlamaConfig",
            "MistralConfig",
            "GemmaConfig",
            "Qwen2Config",
        ]:
            lora_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif lora_modules is None:
            raise ValueError("lora_modules must be specified for this model.")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
        )

        model = get_peft_model(model, config)
        print(f"Model's Lora trainable parameters:")
        model.print_trainable_parameters()
        return model

    def forward(self, inputs): # (b, max_length)
        batch_size = len(inputs)
        input_data = self.encoder.tokenize(inputs)
        input_data = {k: v.cuda() for k, v in input_data.items()}
        embeddings = self.encoder.forward(input_data)
        return embeddings

    # def forward(self, inputs):  # (b, max_length)
    #     input_data = self.encoder.tokenize(inputs)
    #     input_data = {k: v.to(self.encoder.model.device) for k, v in input_data.items()}
        
    #     # Use gradient checkpointing and mixed precision
    #     with autocast():
    #         def forward_fn(input_ids, attention_mask):
    #             return self.encoder.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            
    #         last_hidden_state = checkpoint(forward_fn, input_data['input_ids'], input_data['attention_mask'])
        
    #     # Apply pooling
    #     if self.encoder.pooling_mode == "mean":
    #         embeddings = (last_hidden_state * input_data['attention_mask'].unsqueeze(-1)).sum(1) / input_data['attention_mask'].sum(-1, keepdim=True)
    #     elif self.encoder.pooling_mode == "cls":
    #         embeddings = last_hidden_state[:, 0]
    #     else:
    #         raise ValueError(f"Unsupported pooling mode: {self.encoder.pooling_mode}")
        
    #     return embeddings

        

from llm2vec import LLM2Vec
class EncodingModel_LLM2vec(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
                "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            )
        self.encoder = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            merge_peft=True,
            pooling_mode="mean",
            max_length=512,
            skip_instruction = False,
            
        )
        # merge LoRA of encoder

        
        self.encoder.model = self.initialize_peft(
            self.encoder.model,
        )
        # vector_dim = 768
        # self.vector_linear = nn.Sequential(
        #     nn.Linear(in_features=4096, out_features=vector_dim),
        #     nn.Tanh()
        # ).to('cuda', dtype=torch.bfloat16)
        # set dtype of vector_linear to bfloat16
            
    def initialize_peft(
        self,
        model,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        lora_modules: Optional[List[str]] = None,
    ):
        if lora_modules is None and model.config.__class__.__name__ in [
            "LlamaConfig",
            "MistralConfig",
            "GemmaConfig",
            "Qwen2Config",
        ]:
            lora_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif lora_modules is None:
            raise ValueError("lora_modules must be specified for this model.")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
        )

        model = get_peft_model(model, config)
        print(f"Model's Lora trainable parameters:")
        model.print_trainable_parameters()
        return model

    def forward(self, inputs): # (b, max_length)
        batch_size = len(inputs)
        input_data = self.encoder.tokenize(inputs)
        input_data = {k: v.cuda() for k, v in input_data.items()}
        embeddings = self.encoder.forward(input_data)
        # embeddings = self.vector_linear(embeddings)
        return embeddings
    def forward_mixup(self, inputs):
        """
        inputs: batch of [input1, input2]
        """
        batch_size = len(inputs)
        merged_inputs = []
        for i in range(batch_size):
            merged_inputs.append(inputs[i][0] + " " + inputs[i][1])
        input_data = self.tokenizer(merged_inputs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_data = {k: v.cuda() for k, v in input_data.items()}
        # take tokens from inputs[i][0] and inputs[i][1]
        outputs = self.encoder.model(**input_data)
        last_hidden_state = outputs.last_hidden_state
        temp = [inputs[i][0] for i in range(batch_size)]
        input_data_1 = self.tokenizer(temp)
        len_first = [len(input_data_1['input_ids'][i]) for i in range(batch_size)]

        last_token_indices = input_data['attention_mask'].sum(dim=1) - 1
        last_token_indices = last_token_indices.tolist()
        
        first = [last_hidden_state[i, :len_first[i], :].mean(dim = 0) for i in range(batch_size)]
        second = [last_hidden_state[i, len_first[i]:last_token_indices[i], :].mean(dim = 0) for i in range(batch_size)]
        first = torch.stack(first)
        second = torch.stack(second)
        # compress to bert size
        # first = self.vector_linear(first)
        # second = self.vector_linear(second)
        return first, second

    def compress_to_bert_size(self, vectors):
        return self.vector_linear(vectors)

    def save_lora_weights(self, save_path):
        """
        Save the LoRA weights of the model.
        
        Args:
            save_path (str): The path where the LoRA weights will be saved. (.../encoder)
        """
        # Ensure the encoder model is in evaluation mode
        self.encoder.model.eval()
        
        # Get the underlying PEFT model
        peft_model = self.encoder.model
        
        # Check if the model is a PeftModel
        if not isinstance(peft_model, PeftModel):
            raise ValueError("The model doesn't seem to be a PeftModel. LoRA weights cannot be saved.")
        
        # Save only the LoRA weights
        peft_model.save_pretrained(save_path)
        
        # Save the vector linear weights
        torch.save(self.vector_linear.state_dict(), os.path.join(save_path, "vector_linear.pth"))
        
        print(f"LoRA weights saved to {save_path}")
    
    def load_lora_weights(self, load_path):
        """
        Load the LoRA weights into the model.
        
        Args:
            load_path (str): The path where the LoRA weights are saved.
        """
        # Ensure the encoder model is in evaluation mode
        self.encoder.model.eval()
        
        # Get the underlying PEFT model
        peft_model = self.encoder.model
        
        # Check if the model is a PeftModel
        if not isinstance(peft_model, PeftModel):
            raise ValueError("The model doesn't seem to be a PeftModel. LoRA weights cannot be loaded.")
        
        # Load the LoRA weights
        peft_model.load_adapter(load_path, adapter_name="default")
        # Load the vector linear weights
        self.vector_linear.load_state_dict(torch.load(os.path.join(load_path, "vector_linear.pth")))
        print(f"LoRA weights loaded from {load_path}")
        
        # Optionally, you can set the loaded adapter as active
        peft_model.set_adapter("default")
        
        


    # def forward(self, inputs):  # (b, max_length)
    #     input_data = self.encoder.tokenize(inputs)
    #     input_data = {k: v.to(self.encoder.model.device) for k, v in input_data.items()}
        
    #     # Use gradient checkpointing and mixed precision
    #     with autocast():
    #         def forward_fn(input_ids, attention_mask):
    #             return self.encoder.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            
    #         last_hidden_state = checkpoint(forward_fn, input_data['input_ids'], input_data['attention_mask'])
        
    #     # Apply pooling
    #     if self.encoder.pooling_mode == "mean":
    #         embeddings = (last_hidden_state * input_data['attention_mask'].unsqueeze(-1)).sum(1) / input_data['attention_mask'].sum(-1, keepdim=True)
    #     elif self.encoder.pooling_mode == "cls":
    #         embeddings = last_hidden_state[:, 0]
    #     else:
    #         raise ValueError(f"Unsupported pooling mode: {self.encoder.pooling_mode}")
        
    #     return embeddings

        
        