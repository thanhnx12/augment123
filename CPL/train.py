import argparse
import torch
import random
import sys
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from config import Config


from sampler import data_sampler_CFRL
from data_loader import get_data_loader_BERT
from utils import Moment, gen_data
from encoder import EncodingModel
from add_loss import MultipleNegativesRankingLoss, SupervisedSimCSELoss, ContrastiveLoss
from transformers import BertTokenizer
from mixup import mixup_data_augmentation


class Manager(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
    def _edist(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        L2dist = nn.PairwiseDistance(p=2)
        dist = [] # B
        for i in range(b):
            dist_i = L2dist(x2, x1[i])
            dist.append(torch.unsqueeze(dist_i, 0)) # (N) --> (1,N)
        dist = torch.cat(dist, 0) # (B, N)
        return dist

    def get_memory_proto(self, encoder, dataset):
        '''
        only for one relation data
        '''
        data_loader = get_data_loader_BERT(config, dataset, shuffle=False, \
            drop_last=False,  batch_size=1) 
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)    
        features = torch.cat(features, dim=0) # (M, H)
        proto = features.mean(0)

        return proto, features   

    def select_memory(self, encoder, dataset):
        '''
        only for one relation data
        '''
        N, M = len(dataset), self.config.memory_size
        data_loader = get_data_loader_BERT(self.config, dataset, shuffle=False, \
            drop_last= False, batch_size=1) # batch_size must = 1
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)

        features = np.concatenate(features) # tensor-->numpy array; (N, H)
        
        if N <= M: 
            return copy.deepcopy(dataset), torch.from_numpy(features)

        num_clusters = M # memory_size < len(dataset)
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features) # (N, M)

        mem_set = []
        mem_feas = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            sample = dataset[sel_index]
            mem_set.append(sample)
            mem_feas.append(features[sel_index])

        mem_feas = np.stack(mem_feas, axis=0) # (M, H)
        mem_feas = torch.from_numpy(mem_feas)
        # proto = memory mean
        # rel_proto = mem_feas.mean(0)
        # proto = all mean
        features = torch.from_numpy(features) # (N, H) tensor
        rel_proto = features.mean(0) # (H)

        return mem_set, mem_feas
        # return mem_set, features, rel_proto
    def get_augment_data_label(self, instance, labels):
        max_len = self.config.max_length
        batch_size, dim = instance['ids'].shape

        augmented_ids = []
        augmented_masks = []
        augmented_labels = []
        label_first = []
        label_second = []

        if batch_size < 4:
            random_list_i = random.sample(range(0, batch_size), min(4, batch_size))
            random_list_j = random.sample(range(0, batch_size), min(4, batch_size))
        else:
            random.seed(42)
            random_list_i = random.sample(range(0, batch_size), 4)
            remaining_elements = list(set(range(0, batch_size)) - set(random_list_i))
            
            if len(remaining_elements) >= 4:
                random_list_j = random.sample(remaining_elements, 4)
            else:
                random_list_j = random.sample(range(0, batch_size), 4)

        for i in random_list_i:
            for j in random_list_j:
                # Filter 'ids' using the corresponding 'mask' to remove zero padding
                ids1 = instance['ids'][i][instance['mask'][i] != 0]  # Remove padding from the first sequence
                ids2 = instance['ids'][j][instance['mask'][j] != 0]  # Remove padding from the second sequence

                # Concatenate the filtered sequences
                combined_ids = torch.cat((ids1, ids2)).to(config.device)

                # Truncate the concatenated sequence if it exceeds max_len - 1 and add [102] at the end
                if len(combined_ids) > max_len - 1:
                    combined_ids = combined_ids[:max_len - 1]
                    combined_ids = torch.cat((combined_ids, torch.tensor([102], dtype=combined_ids.dtype).to(config.device))).to(config.device)

                    # Calculate the mask: 1 for valid positions, 0 for padding
                combined_mask = torch.ones_like(combined_ids, dtype=torch.float).to(config.device)

                # Pad with zeros if the sequence is shorter than max_len
                if len(combined_ids) < max_len:
                    padding_length = max_len - len(combined_ids)
                    padding = torch.zeros(padding_length, dtype=combined_ids.dtype).to(config.device)
                    combined_ids = torch.cat((combined_ids, padding)).to(config.device)

                    # Update the mask with zeros for padded positions
                    combined_mask = torch.cat((combined_mask, torch.zeros(padding_length, dtype=torch.float).to(config.device)))

                augmented_ids.append(combined_ids)
                augmented_masks.append(combined_mask)

                # Construct the label pairs
                new_label = torch.tensor([labels[i], labels[j]])
                augmented_labels.append(new_label)
                label_first.append(labels[i])
                label_second.append(labels[j])

        # Convert the lists into tensors
        augmented_data = {
            'ids': torch.stack(augmented_ids),
            'mask': torch.stack(augmented_masks)
        }
        augmented_labels = torch.stack(augmented_labels)
        label_first = torch.tensor(label_first)
        label_second = torch.tensor(label_second)

        return augmented_data, augmented_labels, label_first, label_second
    
    def mask_dropout(self, input_ids, attention_mask, mask_token_id, dropout_rate=0.1):
        """
        Mask dropout implementation of hard prompt
        """
        batch_size, seq_length = input_ids.shape
        dropout_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        for i in range(batch_size):
            # Find the position of the existing [MASK] token
            mask_position = (input_ids[i] == mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_position) == 0:
                mask_position = seq_length
            else:
                mask_position = mask_position[0]
            
            # Apply dropout only to tokens before the [MASK] token
            dropout_mask[i, :mask_position] = torch.bernoulli(torch.full((mask_position,), 1 - dropout_rate)).bool()
        
        # Combine dropout mask with attention mask
        combined_mask = dropout_mask & attention_mask.bool()
        
        # Create masked input ids
        masked_input_ids = input_ids.clone()
        masked_input_ids[~combined_mask] = mask_token_id
        
        # Update attention mask
        new_attention_mask = attention_mask.clone()
        new_attention_mask[~combined_mask] = 0
        
        return masked_input_ids, new_attention_mask
    
    def train_model(self, encoder, training_data, seen_des, is_memory=False):
        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True)
        optimizer = optim.Adam(params=encoder.parameters(), lr=self.config.lr)
        encoder.train()
        epoch = self.config.epoch_mem if is_memory else self.config.epoch
        
        loss_retrieval = MultipleNegativesRankingLoss()
        supervised_simcse_loss = SupervisedSimCSELoss()
        
        
        for i in range(epoch):
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                for k in instance.keys():
                    instance[k] = instance[k].to(self.config.device)
                hidden = encoder(instance)
                
                #-----------------Description-----------------
                # batch_instance = {'ids': [], 'mask': []} 
                # # Create ids and mask tensors in a more compact way
                # batch_instance['ids'] = torch.stack([torch.tensor(seen_des[self.id2rel[label.item()]]['ids']) for label in labels]).to(self.config.device)
                # batch_instance['mask'] = torch.stack([torch.tensor(seen_des[self.id2rel[label.item()]]['mask']) for label in labels]).to(self.config.device)
                # n = len(labels)
                # labels_tensor = torch.tensor(labels).unsqueeze(1)  # Shape: (n, 1)
                # new_matrix_labels_tensor_des = (labels_tensor == labels_tensor.T).to(torch.float).to(self.config.device)
                # hidden_des = encoder.forward_des(batch_instance) # b, dim
                # loss2 = loss_retrieval(hidden, hidden_des, new_matrix_labels_tensor_des)
                #-----------------Description------------------
                
                                
                #-----------------Augment data-----------------
                # augmented_instance, augmented_labels, label_first, label_second = self.get_augment_data_label(instance, labels)
                # mask_hidden_1, mask_hidden_2 = encoder(augmented_instance, is_augment = True)

                # n = len(label_first)
                # m = len(label_second)
                # new_matrix_labels = np.zeros((n, m), dtype=float)

                # # Fill the matrix according to the label comparison
                # for i1 in range(n):
                #     for j in range(m):
                #         if label_first[i1] == label_second[j]:
                #             new_matrix_labels[i1][j] = 1.0

                # new_matrix_labels_tensor = torch.tensor(new_matrix_labels).to(config.device)

                # loss4 = loss_retrieval(mask_hidden_1, mask_hidden_2, new_matrix_labels_tensor)
                #-----------------Augment data-----------------
                
                #-----------------contrastive augment vs des---
                # mask_hidden_mean_12 = (mask_hidden_1 + mask_hidden_2) / 2
                # label2desembedding = {}
                # labels_list = labels.tolist()
                # for i in range(len(labels)):
                #     label2desembedding[labels_list[i]] = hidden_des[i]
                # label_first = label_first.tolist()
                # label_second = label_second.tolist()
                # hidden_des_mean_12 = [(label2desembedding[label_first[i]] + label2desembedding[label_second[i]]) / 2 for i in range(len(label_first))]
                # hidden_des_mean_12 = torch.stack(hidden_des_mean_12, dim=0)
                
                # new_matrix_labels_tensor_des = (labels_tensor == labels_tensor.T).to(torch.float).to(self.config.device)
                # matrix_labels_tensor_mean_12 = np.zeros((mask_hidden_mean_12.shape[0], mask_hidden_mean_12.shape[0]), dtype=float)
                # for i1 in range(mask_hidden_mean_12.shape[0]):
                #     for j1 in range(mask_hidden_mean_12.shape[0]):
                #         if label_first[i1] in [label_first[j1], label_second[j1]] and label_second[i1] in [label_first[j1], label_second[j1]]:
                #             matrix_labels_tensor_mean_12[i1][j1] = 1.0
                # matrix_labels_tensor_mean_12 = torch.tensor(matrix_labels_tensor_mean_12).to(config.device)

                # loss3 = loss_retrieval(mask_hidden_mean_12, mask_hidden_mean_12, matrix_labels_tensor_mean_12)
                #-----------------contrastive augment vs des---
                
                # Add supervised SimCSE loss with mask dropout
                # mask_token_id = self.tokenizer.mask_token_id
                # masked_ids, masked_mask = self.mask_dropout(instance['ids'], instance['mask'], mask_token_id)
                # masked_instance = {'ids': masked_ids, 'mask': masked_mask}
                # masked_hidden = encoder.forward_mask_dropout(masked_instance)
                # simcse_loss = supervised_simcse_loss(hidden, masked_hidden, labels)
                
                
                loss = self.moment.contrastive_loss(hidden, labels, is_memory)
                # print("Losses: ", loss, loss2, loss3, loss4, simcse_loss)
                loss = loss
                # + loss3 + 0.5*loss4
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # update moment
                if is_memory:
                    self.moment.update(ind, hidden.detach().cpu().data, is_memory=True)
                    # self.moment.update_allmem(encoder)
                else:
                    self.moment.update(ind, hidden.detach().cpu().data, is_memory=False)
                # print
                if is_memory:
                    sys.stdout.write('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                else:
                    sys.stdout.write('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                sys.stdout.flush() 
        print('')             
    def train_model_mixup(self, encoder, training_data, seen_des):
        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True)
        optimizer = optim.Adam(params=encoder.parameters(), lr=self.config.lr)
        encoder.train()
        epoch = 1
        
        loss_retrieval = MultipleNegativesRankingLoss()
        supervised_simcse_loss = SupervisedSimCSELoss()
        contrastive_loss = ContrastiveLoss()
        
        
        for i in range(epoch):
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                for k in instance.keys():
                    instance[k] = instance[k].to(self.config.device)
                
                label_first = [temp[0] for temp in labels]
                label_second = [temp[1] for temp in labels]
                
                mask_hidden_1, mask_hidden_2 = encoder.forward_mixup(instance)
                n = len(label_first)
                m = len(label_second)
                new_matrix_labels = np.zeros((n, m), dtype=float)

                # Fill the matrix according to the label comparison
                for i1 in range(n):
                    for j in range(m):
                        if label_first[i1] == label_second[j]:
                            new_matrix_labels[i1][j] = 1.0

                new_matrix_labels_tensor = torch.tensor(new_matrix_labels).to(config.device)
                loss1 = loss_retrieval(mask_hidden_1, mask_hidden_2, new_matrix_labels_tensor)
                
                # mask_hidden_mean_12 = (mask_hidden_1 + mask_hidden_2) / 2
                
                # matrix_labels_tensor_mean_12 = np.zeros((mask_hidden_mean_12.shape[0], mask_hidden_mean_12.shape[0]), dtype=float)
                # for i1 in range(mask_hidden_mean_12.shape[0]):
                #         for j1 in range(mask_hidden_mean_12.shape[0]):
                #             if i1 != j1:
                #                 if label_first[i1] in [label_first[j1], label_second[j1]] and label_second[i1] in [label_first[j1], label_second[j1]]:
                #                     matrix_labels_tensor_mean_12[i1][j1] = 1.0
                # matrix_labels_tensor_mean_12 = torch.tensor(matrix_labels_tensor_mean_12).to(config.device)
                
                # loss2 = loss_retrieval(mask_hidden_mean_12, mask_hidden_mean_12, matrix_labels_tensor_mean_12)
                
                # loss = loss1 + loss2
                
                merged_hidden = torch.cat((mask_hidden_1, mask_hidden_2), dim=0)
                merged_labels = torch.cat((torch.tensor(label_first), torch.tensor(label_second)), dim=0)
                
    
                if merged_hidden.shape[1] != 768: # hard code :)
                    print('something wrong')
                    continue
                loss = self.moment.contrastive_loss(merged_hidden, merged_labels, is_memory = True)
                loss = loss*0.5 + loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.moment.update(ind, mask_hidden_1.detach().cpu().data, is_memory=True)
                # print
                sys.stdout.write('MixupTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                sys.stdout.flush() 
        print('')             
        
    def eval_encoder_proto(self, encoder, seen_proto, seen_relid, test_data):
        batch_size = 16
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)
        
        corrects = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.cpu().data # place in cpu to eval
            logits = -self._edist(fea, seen_proto) # (B, N) ;N is the number of seen relations

            cur_index = torch.argmax(logits, dim=1) # (B)
            pred =  []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size
            sys.stdout.write('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   '\
                .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')
            sys.stdout.flush()        
        print('')
        return corrects / total

    def _get_sample_text(self, data_path, index):
        sample = {}
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i == index:
                    items = line.strip().split('\t')
                    sample['relation'] = self.id2rel[int(items[0])-1]
                    sample['tokens'] = items[2]
                    sample['h'] = items[3]
                    sample['t'] = items[5]
        return sample

    def _read_description(self, r_path):
        rset = {}
        with open(r_path, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                rset[items[1]] = items[2]
        return rset


    def train(self):
        # sampler 
        sampler = data_sampler_CFRL(config=self.config, seed=self.config.seed)
        print('prepared data!')
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.r2desc = self._read_description(self.config.relation_description)

        # encoder
        encoder = EncodingModel(self.config)

        # step is continual task number
        cur_acc, total_acc = [], []
        cur_acc_num, total_acc_num = [], []
        memory_samples = {}
        data_generation = []
        
        seen_des = {}
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        for step, (training_data, valid_data, test_data, current_relations, \
            historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):
            
            for rel in current_relations:
                ids = self.tokenizer.encode(seen_descriptions[rel],
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.config.max_length)        
                # mask
                mask = np.zeros(self.config.max_length, dtype=np.int32)
                end_index = np.argwhere(np.array(ids) == self.tokenizer.get_vocab()[self.tokenizer.sep_token])[0][0]
                mask[:end_index + 1] = 1 
                if rel not in seen_des:
                    seen_des[rel] = {}
                    seen_des[rel]['ids'] = ids
                    seen_des[rel]['mask'] = mask
            
            # Initialization
            self.moment = Moment(self.config)

            # Train current task
            training_data_initialize = []
            for rel in current_relations:
                training_data_initialize += training_data[rel]
            self.moment.init_moment(encoder, training_data_initialize, is_memory=False)
            self.train_model(encoder, training_data_initialize,seen_des)

            # Select memory samples
            for rel in current_relations:
                memory_samples[rel], _ = self.select_memory(encoder, training_data[rel])

            # Data gen
            if self.config.gen == 1:
                gen_text = []
                for rel in current_relations:
                    for sample in memory_samples[rel]:
                        sample_text = self._get_sample_text(self.config.training_data, sample['index'])
                        gen_samples = gen_data(self.r2desc, self.rel2id, sample_text, self.config.num_gen, self.config.gpt_temp, self.config.key)
                        gen_text += gen_samples
                for sample in gen_text:
                    data_generation.append(sampler.tokenize(sample))
                    
            # Train memory
            if step > 0:
                memory_data_initialize = []
                for rel in seen_relations:
                    memory_data_initialize += memory_samples[rel]
                memory_data_initialize += data_generation
                # augment data:
                data_for_train = training_data_initialize + memory_data_initialize
                mixup_samples = mixup_data_augmentation(data_for_train)
                print('Mixup data size: ', len(mixup_samples))
                self.moment.init_moment_mixup(encoder, mixup_samples, is_memory=True) 
                self.train_model_mixup(encoder, mixup_samples, seen_des)
                self.moment.init_moment(encoder, memory_data_initialize, is_memory=True)
                self.train_model(encoder, memory_data_initialize, seen_des, is_memory=True)
                

            # Update proto
            seen_proto = []  
            for rel in seen_relations:
                proto, _ = self.get_memory_proto(encoder, memory_samples[rel])
                seen_proto.append(proto)
            seen_proto = torch.stack(seen_proto, dim=0)

            # get seen relation id
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])
            
            # get representation of seen description
            seen_des_by_id = {}
            for rel in seen_relations:
                seen_des_by_id[self.rel2id[rel]] = seen_des[rel]
            list_seen_des = []
            for i in range(len(seen_proto)):
                list_seen_des.append(seen_des_by_id[seen_relid[i]])

            rep_des = []
            for i in range(len(list_seen_des)):
                sample = {
                    'ids' : torch.tensor([list_seen_des[i]['ids']]).to(self.config.device),
                    'mask' : torch.tensor([list_seen_des[i]['mask']]).to(self.config.device)
                }
                hidden = encoder.forward_des(sample)
                hidden = hidden.detach().cpu().data
                rep_des.append(hidden)
            rep_des = torch.cat(rep_des, dim=0)
            
            # Eval current task and history task
            test_data_initialize_cur, test_data_initialize_seen = [], []
            for rel in current_relations:
                test_data_initialize_cur += test_data[rel]
            for rel in seen_relations:
                test_data_initialize_seen += historic_test_data[rel]
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])
            ac1 = self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_cur)
            ac2 = self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_seen)
            cur_acc_num.append(ac1)
            total_acc_num.append(ac2)
            cur_acc.append('{:.4f}'.format(ac1))
            total_acc.append('{:.4f}'.format(ac2))
            print('cur_acc: ', cur_acc)
            print('his_acc: ', total_acc)

        torch.cuda.empty_cache()
        return total_acc_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="FewRel", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--num_gen", default=2, type=int)
    args = parser.parse_args()
    config = Config('config.ini')
    config.task_name = args.task_name
    config.num_k = args.num_k
    config.num_gen = args.num_gen

    # config 
    print('#############params############')
    print(config.device)
    config.device = torch.device(config.device)
    print(f'Task={config.task_name}, {config.num_k}-shot')
    print(f'Encoding model: {config.model}')
    print(f'pattern={config.pattern}')
    print(f'mem={config.memory_size}, margin={config.margin}, gen={config.gen}, gen_num={config.num_gen}')
    print('#############params############')

    if config.task_name == 'FewRel':
        config.rel_index = './data/CFRLFewRel/rel_index.npy'
        config.relation_name = './data/CFRLFewRel/relation_name.txt'
        config.relation_description = './data/CFRLFewRel/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/test_0.txt'
    else:
        config.rel_index = './data/CFRLTacred/rel_index.npy'
        config.relation_name = './data/CFRLTacred/relation_name.txt'
        config.relation_description = './data/CFRLTacred/relation_description_raw.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_5/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_5/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_10/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_10/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_10/test_0.txt'        

    # seed 
    random.seed(config.seed) 
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)   
    base_seed = config.seed

    acc_list = []
    for i in range(config.total_round):
        config.seed = base_seed + i * 100
        print('--------Round ', i)
        print('seed: ', config.seed)
        manager = Manager(config)
        acc = manager.train()
        acc_list.append(acc)
        torch.cuda.empty_cache()
    
    accs = np.array(acc_list)
    ave = np.mean(accs, axis=0)
    print('----------END')
    print('his_acc mean: ', np.around(ave, 4))



            
        
            
            

