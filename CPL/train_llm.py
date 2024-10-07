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


from sampler_bert_llm import data_sampler_CFRL
from data_loader import get_data_loader_BERTLLM
from utils_llm import Moment, gen_data
from encoder_llm import EncodingModel_LLM2vec
from add_loss import MultipleNegativesRankingLoss, SupervisedSimCSELoss, ContrastiveLoss, NegativeCosSimLoss
from transformers import BertTokenizer
from mixup import mixup_data_augmentation_llm
from sam import SAM
import logging


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
        data_loader = get_data_loader_BERTLLM(config, dataset, shuffle=False, \
            drop_last=False,  batch_size=1) 
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            hidden = encoder(instance['input']).float()
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
        data_loader = get_data_loader_BERTLLM(self.config, dataset, shuffle=False, \
            drop_last= False, batch_size=1) # batch_size must = 1
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            hidden = encoder(instance['input']).float()
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
        
    def train_model(self, encoder, training_data, is_memory=False):
        data_loader = get_data_loader_BERTLLM(self.config, training_data, shuffle=True)
        optimizer = optim.Adam(params=encoder.parameters(), lr=self.config.lr)
        if self.config.SAM:
            base_optimizer = optim.Adam
            optimizer = SAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, adaptive=True, lr=self.config.lr)
        encoder.train()
        epoch = self.config.epoch_mem if is_memory else self.config.epoch

        for i in range(epoch):
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                hidden = encoder(instance['input'])
                loss = self.moment.contrastive_loss(hidden, labels, is_memory)    
                if not self.config.SAM:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    hidden = encoder(instance['input'])
                    loss = self.moment.contrastive_loss(hidden, labels, is_memory)
                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                    # update moment
                if is_memory:
                    self.moment.update(ind, hidden.detach().cpu().data, is_memory=True)
                    # self.moment.update_allmem(encoder)
                else:
                    self.moment.update(ind, hidden.detach().cpu().data, is_memory=False)
                # print
                if is_memory:
                    sys.stdout.write('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                    # logger.info('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()))
                else:
                    sys.stdout.write('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                    # logger.info('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()))
                sys.stdout.flush() 
        print('')           
    def train_model_mixup(self, encoder, training_data):
        data_loader = get_data_loader_BERTLLM(self.config, training_data, shuffle=True)
        optimizer = optim.Adam(params=encoder.parameters(), lr=self.config.lr)
        if self.config.SAM:
            base_optimizer = optim.Adam
            optimizer = SAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, adaptive=True, lr=self.config.lr)
        encoder.train()
        epoch = 1
        
        loss_retrieval = MultipleNegativesRankingLoss()
        neg_cos_sim_loss = NegativeCosSimLoss()
        
        
        for i in range(epoch):
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                label_first = [temp[0] for temp in labels]
                label_second = [temp[1] for temp in labels]
                
                mask_hidden_1, mask_hidden_2 = encoder.forward_mixup(instance['input'])
                n = len(label_first)
                m = len(label_second)
                new_matrix_labels = np.zeros((n, m), dtype=float)

                # Fill the matrix according to the label comparison
                for i1 in range(n):
                    for j in range(m):
                        if label_first[i1] == label_second[j]:
                            new_matrix_labels[i1][j] = 1.0

                new_matrix_labels_tensor = torch.tensor(new_matrix_labels).to(config.device)
                loss1 = neg_cos_sim_loss(mask_hidden_1, mask_hidden_2)
                
                mask_hidden_mean_12 = (mask_hidden_1 + mask_hidden_2) / 2
                
                matrix_labels_tensor_mean_12 = np.zeros((mask_hidden_mean_12.shape[0], mask_hidden_mean_12.shape[0]), dtype=float)
                for i1 in range(mask_hidden_mean_12.shape[0]):
                        for j1 in range(mask_hidden_mean_12.shape[0]):
                            if i1 != j1:
                                if label_first[i1] in [label_first[j1], label_second[j1]] and label_second[i1] in [label_first[j1], label_second[j1]]:
                                    matrix_labels_tensor_mean_12[i1][j1] = 1.0
                matrix_labels_tensor_mean_12 = torch.tensor(matrix_labels_tensor_mean_12).to(config.device)
                
                loss2 = loss_retrieval(mask_hidden_mean_12, mask_hidden_mean_12, matrix_labels_tensor_mean_12)
                
                
                merged_hidden = torch.cat((mask_hidden_1, mask_hidden_2), dim=0)
                merged_labels = torch.cat((torch.tensor(label_first), torch.tensor(label_second)), dim=0)
                
    
                if merged_hidden.shape[1] != 4096: # hard code :)
                    print('something wrong')
                    logger.info('something wrong')
                    continue
                loss = self.moment.contrastive_loss(merged_hidden, merged_labels, is_memory = True)
                sum_loss = 0.0
                if not torch.isnan(loss1).any():
                    sum_loss += self.config.mixup_loss_1*loss1
                if not torch.isnan(loss2).any():
                    sum_loss += self.config.mixup_loss_2*loss2
                if not torch.isnan(loss).any():
                    sum_loss += 0.5*loss
                
                if not self.config.SAM:
                    optimizer.zero_grad()
                    sum_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    mask_hidden_1, mask_hidden_2 = encoder.forward_mixup(instance['input'])
                    loss1 = neg_cos_sim_loss(mask_hidden_1, mask_hidden_2)
                    mask_hidden_mean_12 = (mask_hidden_1 + mask_hidden_2) / 2
                    loss2 = loss_retrieval(mask_hidden_mean_12, mask_hidden_mean_12, matrix_labels_tensor_mean_12)
                    
                    
                    merged_hidden = torch.cat((mask_hidden_1, mask_hidden_2), dim=0)
                    merged_labels = torch.cat((torch.tensor(label_first), torch.tensor(label_second)), dim=0)
                    
        
                    if merged_hidden.shape[1] != 4096: # hard code :)
                        print('something wrong')
                        logger.info("something wrong")
                        continue
                    loss = self.moment.contrastive_loss(merged_hidden, merged_labels, is_memory = True)
                    sum_loss = 0.0
                    if not torch.isnan(loss1).any():
                        sum_loss += self.config.mixup_loss_1*loss1
                    if not torch.isnan(loss2).any():
                        sum_loss += self.config.mixup_loss_2*loss2
                    if not torch.isnan(loss).any():
                        sum_loss += 0.5*loss
                    sum_loss.backward()
                    optimizer.second_step(zero_grad=True)
                    
                        
                self.moment.update(ind, mask_hidden_1.detach().cpu().data, is_memory=True)
                sys.stdout.write('MixupTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, sum_loss.item()) + '\r')
                # logger.info('MixupTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, sum_loss.item()))
                sys.stdout.flush() 
        print('')             
          
    
    def eval_encoder_proto(self, encoder, seen_proto, seen_relid, test_data):
        batch_size = self.config.batch_size
        test_loader = get_data_loader_BERTLLM(self.config, test_data, False, False, batch_size)
        
        corrects = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            hidden = encoder(instance['input']).float()
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
        logger.info('prepared data!')
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.r2desc = self._read_description(self.config.relation_description)

        # encoder
        encoder = EncodingModel_LLM2vec(self.config)

        # step is continual task number
        cur_acc, total_acc = [], []
        cur_acc_num, total_acc_num = [], []
        memory_samples = {}
        data_generation = []
        
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        for step, (training_data, valid_data, test_data, current_relations, \
            historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):
            
            # Initialization
            self.moment = Moment(self.config)

            # Train current task
            training_data_initialize = []
            for rel in current_relations:
                training_data_initialize += training_data[rel]
            if self.config.SAM_type == 'current':
                self.config.SAM = True
            if self.config.SAM_type == 'full' :
                self.config.SAM = True
            self.moment.init_moment(encoder, training_data_initialize, is_memory=False)
            self.train_model(encoder, training_data_initialize)
            if self.config.SAM_type == 'current':
                self.config.SAM = False

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
                if config.mixup:
                    mixup_samples = mixup_data_augmentation_llm(data_for_train)
                    print('Mixup data size: ', len(mixup_samples))
                    self.moment.init_moment_mixup(encoder, mixup_samples, is_memory=True) 
                    self.train_model_mixup(encoder, mixup_samples)
                self.moment.init_moment(encoder, memory_data_initialize, is_memory=True)
                self.train_model(encoder, memory_data_initialize, is_memory=True)
                

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
            logger.info(f"cur_acc: {cur_acc}")
            logger.info(f"his_acc: {total_acc}")

        torch.cuda.empty_cache()
        return total_acc_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="FewRel", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--num_gen", default=2, type=int)
    parser.add_argument("--mixup", action = 'store_true', default=False)
    parser.add_argument("--epoch", default=8, type=int)
    parser.add_argument("--epoch_mem", default=6, type=int)
    parser.add_argument("--mixup_loss_1", default=0.25, type=float)
    parser.add_argument("--mixup_loss_2", default=0.25, type=float)
    parser.add_argument("--SAM", action = 'store_true', default=False)
    parser.add_argument("--SAM_type", default="", type=str, help="current for SAM in current task or full for SAM in all data")
    parser.add_argument("--rho", default=0.05, type=float)
    
    args = parser.parse_args()
    config = Config('config_llm.ini')
    config.task_name = args.task_name
    config.num_k = args.num_k
    config.num_gen = args.num_gen
    config.mixup = args.mixup
    config.epoch = args.epoch
    config.epoch_mem = args.epoch_mem
    config.mixup_loss_1 = args.mixup_loss_1
    config.mixup_loss_2 = args.mixup_loss_2
    config.SAM = args.SAM
    config.SAM_type = args.SAM_type
    config.rho = args.rho

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

    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    
    pre = ""
    if args.mixup: pre += "mixup|"
    if args.SAM: pre += "SAM"

    file_handler = logging.FileHandler(f'CPL-LLM-{pre}-logs-task_{config.task_name}-shot_{config.num_k}-numgen_{config.num_gen}-epoch_{config.epoch}_{config.epoch_mem}-lossfactor_{config.mixup_loss_1}_{config.mixup_loss_2}-rho_{config.rho}-SAM_type_{config.SAM_type}-lr_{config.lr}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger.info('#############params############')
    logger.info(f'Task={config.task_name}, {config.num_k}-shot')
    logger.info(f'Encoding model: {config.model}')
    logger.info(f'pattern={config.pattern}')
    logger.info(f'mem={config.memory_size}, margin={config.margin}, gen={config.gen}, gen_num={config.num_gen}')
    logger.info('#############params############')


    
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
        logger.info(f"--------Round {i}")
        logger.info(f"seed: {config.seed}")
        manager = Manager(config)
        acc = manager.train()
        acc_list.append(acc)
        torch.cuda.empty_cache()
    
    accs = np.array(acc_list)
    ave = np.mean(accs, axis=0)
    print('----------END')
    print('his_acc mean: ', np.around(ave, 4))
    logger.info('----------END')
    logger.info(f'his_acc mean: {np.around(ave, 4)}')





            
        
            
            

