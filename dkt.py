import os 
import numpy as np
import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics


class DKT(Module):
    '''
        Args: 
            num_q: total number of questions in given the dataset
    '''
    def __init__(self, num_q, emb_size, hidden_size):
        super(DKT, self).__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size) 
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, q, r):
        '''
            q: the questions(KC) sequence with the size of [batch_size, n]
            r: response sequence with the size of [batch_size, n]
        '''

        # Compressive sensing
        
        x = q + r 
        h, _ = self.lstm_layer(self.interaction_emb(x))
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y


def dkt_train(model, train_loader, test_loader, exp_loader, num_q, num_epochs, opt, ckpt_path):
    '''
        Args:
            train_loader: the PyTorch DataLoader instance for training
            test_loader: the PyTorch DataLoader instance for test
            num_epochs: the number of epochs
            opt: the optimization to train this model
            ckpt_path: the path to save this model's parameters
    '''
    aucs = []
    loss_means = []  

    max_auc = 0

    for i in range(0, num_epochs):
        loss_mean = []
        

        for data in train_loader:
            # q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, bert_sentences, bert_sentence_types, bert_sentence_att_mask, proc_atshft_sentences
            q, r, qshft_seqs, rshft_seqs, m = data
            model.train()

            y = model(q.long(), r.long())
            y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)

            opt.zero_grad()
            y = torch.masked_select(y, m)
            t = torch.masked_select(rshft_seqs, m)

            loss = binary_cross_entropy(y, t) 
            loss.backward()
            opt.step()

            loss_mean.append(loss.detach().cpu().numpy())

    with torch.no_grad():
        for data in test_loader:
            q, r, qshft_seqs, rshft_seqs, m = data

            model.eval()
            
            y = model(q.long(), r.long())
            y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)

            y = torch.masked_select(y, m).detach().cpu()
            t = torch.masked_select(rshft_seqs, m).detach().cpu()

            auc = metrics.roc_auc_score(
                y_true=t.numpy(), y_score=y.numpy()
            )

            loss_mean = np.mean(loss_mean) 
            
            if auc > max_auc : 
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        ckpt_path, "model.ckpt"
                    )
                )
                print(f"Epoch {i}, previous AUC: {max_auc}, max AUC: {auc}")
                max_auc = auc

            loss_means.append(loss_mean)

    model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.ckpt")))
    for i in range(1, num_epochs + 1):
        with torch.no_grad():
            for data in exp_loader:
                q, r, qshft_seqs, rshft_seqs, m = data

                model.eval()
                y = model(q.long(), r.long())
                y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)


                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(rshft_seqs, m).detach().cpu()

                auc = metrics.roc_auc_score(
                    y_true=t.numpy(), y_score=y.numpy()
                )

                loss_mean = np.mean(loss_mean) 
                
                print(f"Epoch: {i}, AUC: {auc}, Loss Mean: {loss_mean}")

                aucs.append(auc)

    return aucs, loss_means