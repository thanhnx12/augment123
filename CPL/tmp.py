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
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")


from sampler import data_sampler_CFRL
from data_loader import get_data_loader_BERT
from utils import Moment, gen_data
from encoder import EncodingModel
# import wandb

from transformers import BertTokenizer
from add_loss import MultipleNegativesRankingLoss, BatchHardSoftMarginTripletLoss, OnlineContrastiveLoss

class Manager(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(1536, 256).to(config.device)
        self.fc2 = nn.Linear(256, 1).to(config.device)
        
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
    def _cosine_similarity(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        cos = nn.CosineSimilarity(dim=1)
        sim = []
        for i in range(b):
            sim_i = cos(x2, x1[i])
            sim.append(torch.unsqueeze(sim_i, 0))
        sim = torch.cat(sim, 0)
        return sim
    

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
        

    def train_model(self, encoder, training_data, seen_des, is_memory=False):
        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True)
        optimizer = optim.Adam(params=encoder.parameters(), lr=self.config.lr)
        encoder.train()
        epoch = self.config.epoch_mem if is_memory else self.config.epoch
        # softmax = nn.Softmax(dim=0)
        soft_margin_loss = BatchHardSoftMarginTripletLoss()

        for i in range(epoch):
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                for k in instance.keys():
                    instance[k] = instance[k].to(self.config.device)

                batch_instance = {'ids': [], 'mask': []} 

                # for label in labels:
                    # batch_instance['ids'] = torch.tensor([item[0]['ids'] for item in data])
                    # batch_instance['mask'] = torch.tensor([item[0]['mask'] for item in data])

                batch_instance['ids'] = torch.tensor([seen_des[self.id2rel[label.item()]]['ids'] for label in labels]).to(self.config.device)
                batch_instance['mask'] = torch.tensor([seen_des[self.id2rel[label.item()]]['mask'] for label in labels]).to(self.config.device)

                n = len(labels)
                new_matrix_labels = np.zeros((n, n), dtype=float)

                # Fill the matrix according to the label comparison
                for i1 in range(n):
                    for j in range(n):
                        if labels[i1] == labels[j]:
                            new_matrix_labels[i1][j] = 1.0

                new_matrix_labels_tensor = torch.tensor(new_matrix_labels).to(config.device)
                
                # print(new_matrix_labels_tensor)

                    # batch_instance['ids'] = torch.tensor([item[0]['ids'] for item in seen_des[self.id2rel[label.items()]]])
                    # seen_des[self.id2rel[label.items()]]
                    # cur_des.appendseen_des[self.id2rel[label.items()]]

                # labels tensor shape b*b
                
                
                hidden = encoder(instance) # b, dim
                loss = self.moment.contrastive_loss(hidden, labels, is_memory)
                labels_des = encoder(batch_instance, is_des = True) # b, dim

                # compute online contrastive loss
                online_contrastive_loss = OnlineContrastiveLoss()
                loss_online = []
                for idx in range(len(labels)):
                    rep_des = labels_des[idx]
                    label_for_loss = labels == labels[idx]
                    # loss_online += OnlineContrastiveLoss(rep_des, hidden, label_for_loss)
                    loss_online.append(online_contrastive_loss(rep_des, hidden, label_for_loss))
                loss_online = torch.stack(loss_online).mean()




                loss_retrieval = MultipleNegativesRankingLoss()
                loss2 = loss_retrieval(hidden, labels_des, new_matrix_labels_tensor)


                # anchor chính là cái token mask , mấy cái defination là positive negative 

                # compute infonceloss
                # infoNCE_loss = 0
                # list_labels = labels.cpu().numpy().tolist()

                # for j in range(len(list_labels)):
                #     negative_sample_indexs = np.where(np.array(list_labels) != list_labels[j])[0]
                    
                #     positive_hidden = hidden[j].unsqueeze(0)
                #     negative_hidden = hidden[negative_sample_indexs]

                #     positive_lmhead_output = labels_des[j].unsqueeze(0)

                #     # f_pos = encoder.infoNCE_f(positive_hidden, positive_hidden)
                #     # f_neg = encoder.infoNCE_f(negative_hidden, negative_hidden)
                #     # print(positive_hidden.shape) # 1,768
                #     # print(negative_hidden.shape) # 11,768 

                #     f_pos = torch.matmul(positive_lmhead_output, positive_hidden.T)  # Shape: (1, 1)
                #     f_neg = torch.matmul(positive_lmhead_output, negative_hidden.T)  # Shape: (1, N)

                #     # print(f_pos.shape)
                #     # print(f_neg.shape)
                #     f_concat = torch.cat([f_pos, f_neg], dim=1).squeeze()

                #     f_concat = torch.log(torch.max(f_concat , torch.tensor(1e-9).to(self.config.device)))
                #     # print(f_concat.shape)
                #     try:
                #         infoNCE_loss += -torch.log(F.softmax(f_concat)[0])
                #         # print(F.softmax(f_concat, dim = 0)[0].shape)
                #         # print(infoNCE_loss.shape)

                #     except:
                #         None

                # infoNCE_loss = infoNCE_loss / len(list_labels)


                # compute soft margin triplet loss
                uniquie_labels = labels.unique()
                if len(uniquie_labels) > 1:
                    loss3 = soft_margin_loss(hidden, labels.to(self.config.device))
                else:
                    loss3 = 0.0
                
                # cross encoder
                scores = []
                b = hidden.shape[0]
                for i5 in range(b):
                    for j5 in range(b):
                        pair = torch.cat([hidden[i5], labels_des[j5]], dim=0).to(config.device)  # Concatenate along the feature dimension
                        pair = pair.unsqueeze(0)  # Add batch dimension
                        score = F.relu(self.fc1(pair))
                        score = torch.sigmoid(self.fc2(score))
                        # score = cross_encoder(pair)  # Forward pass through cross-encoder
                        scores.append(score)

                # Reshape scores to (b, b)
                new_matrix_labels_tensor = new_matrix_labels_tensor.float()

                scores = torch.cat(scores).view(b, b)
                criterion = nn.BCELoss()
                loss_cross_encoder = criterion(scores, new_matrix_labels_tensor)

                # cross encoder


                loss = loss + loss2 + loss_online + loss3 + loss_cross_encoder
            
                
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
    def eval_encoder_proto_des(self, encoder, seen_proto, seen_relid, test_data, rep_des):
        """
        Args:
            encoder: Encoder
            seen_proto: seen prototypes. NxH tensor
            seen_relid: relation id of protoytpes
            test_data: test data
            rep_des: representation of seen relation description. N x H tensor

        Returns:

        """
        batch_size = 16
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)

        corrects = 0.0
        corrects1 = 0.0
        corrects2 = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.cpu().data  # place in cpu to eval
            logits = -self._edist(fea, seen_proto)  # (B, N) ;N is the number of seen relations
            logits_des = self._cosine_similarity(fea, rep_des)  # (B, N)
#             logits = logits*(1 + logits_des)
            # combine using rrf
            rrf_k = 60
            
            logits_ranks = torch.argsort(torch.argsort(-logits, dim=1), dim=1) + 1
            logits_des_ranks = torch.argsort(torch.argsort(-logits_des, dim=1), dim=1) + 1
            rrf_logits = 0.4 / (rrf_k + logits_ranks)
            rrf_logits_des = 0.6 / (rrf_k + logits_des_ranks)
            logits_rrf = rrf_logits + rrf_logits_des
            # logits = 0.7*rrf_logits + 0.3*rrf_logits_des
            
            # logits = logits + logits_des
            
            cur_index = torch.argmax(logits, dim=1)  # (B)
            pred = []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size

            # by logits_des
            cur_index1 = torch.argmax(logits_des,dim=1)
            pred1 = []
            for i in range(cur_index1.size()[0]):
                pred1.append(seen_relid[int(cur_index1[i])])
            pred1 = torch.tensor(pred1)
            correct1 = torch.eq(pred1, label).sum().item()
            acc1 = correct1/ batch_size
            corrects1 += correct1

            # by rrf
            cur_index2 = torch.argmax(logits_rrf,dim=1)
            pred2 = []
            for i in range(cur_index2.size()[0]):
                pred2.append(seen_relid[int(cur_index2[i])])
            pred2 = torch.tensor(pred2)
            correct2 = torch.eq(pred2, label).sum().item()
            acc2 = correct2/ batch_size
            corrects2 += correct2

            

            sys.stdout.write('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
                             .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')
            sys.stdout.write('[EVAL DES] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
                             .format(batch_num, 100 * acc1, 100 * (corrects1 / total)) + '\r')
            sys.stdout.write('[EVAL RRF] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
                             .format(batch_num, 100 * acc2, 100 * (corrects2 / total)) + '\r')
            sys.stdout.flush()
        print('')
        return corrects / total, corrects1 / total, corrects2 / total

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
        self.config.vocab_size = sampler.config.vocab_size

        print('prepared data!')
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.r2desc = self._read_description(self.config.relation_description)

        # encoder
        encoder = EncodingModel(self.config)

        # step is continual task number
        cur_acc, total_acc = [], []
        cur_acc1, total_acc1 = [], []
        cur_acc2, total_acc2 = [], []


        cur_acc_num, total_acc_num = [], []
        cur_acc_num1, total_acc_num1 = [], []
        cur_acc_num2, total_acc_num2 = [], []


        memory_samples = {}
        data_generation = []
        seen_des = {}


        self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.unused_token = '[unused0]'
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, \
            additional_special_tokens=[self.unused_token])
        # ids = self.tokenizer.encode(' '.join(prompt),
        #                             padding='max_length',
        #                             truncation=True,
        #                             max_length=self.max_length)        
        # # mask
        # mask = np.zeros(self.max_length, dtype=np.int32)
        # end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        # mask[:end_index + 1] = 1 

    



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
            self.train_model(encoder, training_data_initialize, seen_des)

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
                        print(gen_samples)
                        gen_text += gen_samples
                for sample in gen_text:
                    data_generation.append(sampler.tokenize(sample))
                    
            # Train memory
            if step > 0:
                memory_data_initialize = []
                for rel in seen_relations:
                    memory_data_initialize += memory_samples[rel]
                memory_data_initialize += data_generation
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
                hidden = encoder(sample, is_des=True)
                hidden = hidden.detach().cpu().data
                rep_des.append(hidden)
            rep_des = torch.cat(rep_des, dim=0)

            # Eval current task and history task
            test_data_initialize_cur, test_data_initialize_seen = [], []
            for rel in current_relations:
                test_data_initialize_cur += test_data[rel]
            for rel in seen_relations:
                test_data_initialize_seen += historic_test_data[rel]
            # ac1 = self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_cur)
            # ac2 = self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_seen)
            ac1,ac1_des, ac1_rrf = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_cur,rep_des)
            ac2,ac2_des, ac2_rrf = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_seen,rep_des)
            
            cur_acc_num.append(ac1)
            total_acc_num.append(ac2)
            cur_acc.append('{:.4f}'.format(ac1))
            total_acc.append('{:.4f}'.format(ac2))
            print('cur_acc: ', cur_acc)
            print('his_acc: ', total_acc)

            cur_acc_num1.append(ac1_des)
            total_acc_num1.append(ac2_des)
            cur_acc1.append('{:.4f}'.format(ac1_des))
            total_acc1.append('{:.4f}'.format(ac2_des))
            print('cur_acc des: ', cur_acc1)
            print('his_acc des: ', total_acc1)

            cur_acc_num2.append(ac1_rrf)
            total_acc_num2.append(ac2_rrf)
            cur_acc2.append('{:.4f}'.format(ac1_rrf))
            total_acc2.append('{:.4f}'.format(ac2_rrf))
            print('cur_acc rrf: ', cur_acc2)
            print('his_acc rrf: ', total_acc2)


        torch.cuda.empty_cache()
        # save model
        # torch.save(encoder.state_dict(), "./checkpoints/encoder.pth")
        return total_acc_num, total_acc_num1, total_acc_num2


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

    # wandb.init(
    # project = 'CPL',
    # name = f"CPL{args.task_name}_{args.num_k}-shot",
    # config = {
    #     'name': "CPL",
    #     "task" : args.task_name,
    #     "shot" : f"{args.num_k}-shot"
    # }
        # )
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
        config.relation_description = './data/CFRLFewRel/relation_description_detail.txt'
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
        config.relation_description = './data/CFRLTacred/relation_description_detail.txt'
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
    acc_list1 = []
    aac_list2 = []
    for i in range(config.total_round):
        config.seed = base_seed + i * 100
        print('--------Round ', i)
        print('seed: ', config.seed)
        manager = Manager(config)
        acc, acc1, aac2 = manager.train()
        acc_list.append(acc)
        acc_list1.append(acc1)
        aac_list2.append(aac2)
        torch.cuda.empty_cache()
    
    accs = np.array(acc_list)
    ave = np.mean(accs, axis=0)
    print('----------END')
    print('his_acc mean: ', np.around(ave, 4))
    accs1 = np.array(acc_list1)
    ave1 = np.mean(accs1, axis=0)
    print('his_acc des mean: ', np.around(ave1, 4))
    accs2 = np.array(aac_list2)
    ave2 = np.mean(accs2, axis=0)
    print('his_acc rrf mean: ', np.around(ave2, 4))
    