import os
import pickle

import numpy as np 
import pandas as pd 

from torch.utils.data import Dataset
from utils import match_seq_len

DATASET_DIR = "datasets/ASSIST2009/"
Q_SEQ_PICKLE = "q_seqs.pkl"
R_SEQ_PICKLE = "r_seqs.pkl"
AT_SEQ_PICKLE = "at_seqs.pkl"
Q_LIST_PICKLE = "q_list.pkl"
U_LIST_PICKLE = "u_list.pkl"
Q_IDX_PICKLE = "q2idx.pkl"
Q_DIFF_PICKLE = 'q2diff.pkl'
P_ID_PICKLE = 'pid.pkl'
P_LIST_PICKLE = "p_list.pkl"
HINT_LIST_PICKLE = "hint_use.pkl"

class ASSIST2009(Dataset):
    def __init__(self, seq_len, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.dataset_path = os.path.join(
            self.dataset_dir, "skill_builder_data.csv"
        )

        # If It's saved by pickle, loading that file.
        if os.path.exists(os.path.join(self.dataset_dir, Q_SEQ_PICKLE)):
            with open(os.path.join(self.dataset_dir, Q_SEQ_PICKLE), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, R_SEQ_PICKLE), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, Q_LIST_PICKLE), "rb") as f:
                self.q_list = pickle.load(f)
        else:
            self.q_seqs, self.r_seqs, self.q_list = self.preprocess()
            
        # save to number of question.
        self.num_q = self.q_list.shape[0]

        if seq_len:
            self.q_seqs, self.r_seqs = match_seq_len(self.q_seqs, self.r_seqs, seq_len)

        self.len = len(self.q_seqs)
    
    def __getitem__(self, index) :
        return self.q_seqs[index], self.r_seqs[index]
    
    def __len__(self):
        return self.len

    def preprocess(self):
        # We select the feature skill_name instead of question_id. so, it has to remove NULL value.
        df = pd.read_csv(self.dataset_path, encoding='ISO-8859-1').dropna(subset=["skill_name"])\
            .drop_duplicates(subset=["order_id", "skill_name"])\
            .sort_values(by=["order_id"])

        # extract unique user and unique skill_name for knowledge tracing.
        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["skill_name"].values) 

        q2idx = {q: idx for idx, q in enumerate(q_list)}

        q_seqs = []
        r_seqs = []
        
        for u in u_list:
            # search user to sequantial datasets.
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["skill_name"]]) # user has the problem sequence that is solved by userself. 유저의 스킬에 대한 해당 스킬의 인덱스 리스트를 np.array로 형변환
            r_seq = df_u["correct"].values # user's correctness.
            
            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
        with open(os.path.join(self.dataset_dir, Q_SEQ_PICKLE), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, R_SEQ_PICKLE), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, Q_LIST_PICKLE), "wb") as f:
            pickle.dump(q_list, f)
        return q_seqs, r_seqs, q_list