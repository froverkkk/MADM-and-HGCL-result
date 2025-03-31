import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import scipy.sparse as sp
import torch
from utils import normalize_dense, normalize_sp


def load_data_from_files(dataset_name):

    links_file = f"../data/{dataset_name}/{dataset_name}.links"
    rating_file = f"../daya/{dataset_name}/{dataset_name}.rating"




    history_u_lists = {}
    social_adj_lists = {}

    user_ids = set()
    item_ids = set()

    try:
        with open(rating_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user_id = int(parts[0])
                    item_id = int(parts[1])

                    user_ids.add(user_id)
                    item_ids.add(item_id)

                    if user_id not in history_u_lists:
                        history_u_lists[user_id] = []
                    history_u_lists[user_id].append(item_id)

    except Exception as e:

        raise

    social_user_ids = set()

    try:
        with open(links_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user_id = int(parts[0])
                    friend_id = int(parts[1])

                    social_user_ids.add(user_id)
                    social_user_ids.add(friend_id)

                    if user_id not in social_adj_lists:
                        social_adj_lists[user_id] = []
                    social_adj_lists[user_id].append(friend_id)

    except Exception as e:

        raise


    all_user_ids = user_ids.union(social_user_ids)


    user_id_mapping = {}
    next_id = 1
    for uid in sorted(user_ids):
        user_id_mapping[uid] = next_id
        next_id += 1

    num_of_target_users = next_id - 1

    for uid in sorted(social_user_ids - user_ids):
        user_id_mapping[uid] = next_id
        next_id += 1

    num_of_all_users = next_id - 1


    item_id_mapping = {iid: num_of_all_users + i + 1 for i, iid in enumerate(sorted(item_ids))}


    adjusted_history_u_lists = {}
    for uid, items in history_u_lists.items():
        new_uid = user_id_mapping[uid]
        new_items = [item_id_mapping[iid] for iid in items]
        adjusted_history_u_lists[new_uid] = new_items

    adjusted_social_adj_lists = {}
    for uid, friends in social_adj_lists.items():
        if uid in user_id_mapping:
            new_uid = user_id_mapping[uid]
            new_friends = [user_id_mapping[fid] for fid in friends if fid in user_id_mapping]
            if new_friends:
                adjusted_social_adj_lists[new_uid] = new_friends


    num_of_items = len(item_ids)
    num_of_nodes = num_of_all_users + num_of_items


    avg_interaction = sum(len(items) for items in adjusted_history_u_lists.values()) / len(
        adjusted_history_u_lists) if adjusted_history_u_lists else 0
    avg_friend = sum(len(friends) for friends in adjusted_social_adj_lists.values()) / len(
        adjusted_social_adj_lists) if adjusted_social_adj_lists else 0


    user_collab = None



    return (adjusted_history_u_lists, None, adjusted_social_adj_lists, None, user_collab,
            avg_interaction, avg_friend, num_of_target_users, num_of_all_users, num_of_nodes)


def get_spRRT(config, args, rating_valid, rating_test):

    n_users = config['n_users']
    n_items = config['n_items']
    all_ratings = config['user_rating']


    sp_uu_collab_adj_mat_tr, sp_uu_collab_adj_mat_val = create_spRRT(n_users, n_items, all_ratings, rating_valid,
                                                                     rating_test)

    return sp_uu_collab_adj_mat_tr, sp_uu_collab_adj_mat_val


def create_spRRT(n_users, n_items, all_ratings, rating_valid, rating_test):
    R_tr = np.zeros((n_users, n_items))
    R_val = np.zeros((n_users, n_items))
    for uid in all_ratings.keys():
        for item in all_ratings[uid]:
            if item not in rating_test[uid]:
                R_val[uid-1, item-n_users-1] = 1
                if item not in rating_valid[uid]:
                    R_tr[uid-1, item-n_users-1] = 1
    uu_collab_adj_mat_tr = np.dot(R_tr, R_tr.T)
    row, col = np.diag_indices_from(uu_collab_adj_mat_tr)
    uu_collab_adj_mat_tr[row, col] = 1
    uu_collab_adj_mat_tr = sp.dok_matrix(uu_collab_adj_mat_tr)

    uu_collab_adj_mat_val = np.dot(R_val, R_val.T)
    uu_collab_adj_mat_val[row, col] = 1
    uu_collab_adj_mat_val = sp.dok_matrix(uu_collab_adj_mat_val)
    
    return uu_collab_adj_mat_tr.tocsr(), uu_collab_adj_mat_val.tocsr()


def get_RRT(config, args, rating_valid, rating_test):

    n_users = config['n_users']
    n_items = config['n_items']
    all_ratings = config['user_rating']


    uu_collab_adj_mat_tr, uu_collab_adj_mat_val = create_RRT(n_users, n_items, all_ratings, rating_valid, rating_test)

    return uu_collab_adj_mat_tr, uu_collab_adj_mat_val




def create_RRT(n_users, n_items, all_ratings, rating_valid, rating_test):
    R_tr = np.zeros((n_users, n_items))
    R_val = np.zeros((n_users, n_items))
    for uid in all_ratings.keys():
        for item in all_ratings[uid]:
            if item not in rating_test[uid]:
                R_val[uid-1, item-n_users-1] = 1
                if item not in rating_valid[uid]:
                    R_tr[uid-1, item-n_users-1] = 1
    uu_collab_adj_mat_tr = np.dot(R_tr, R_tr.T)
    row, col = np.diag_indices_from(uu_collab_adj_mat_tr)
    uu_collab_adj_mat_tr[row, col] = 1
    uu_collab_adj_mat_tr = normalize_dense(uu_collab_adj_mat_tr)

    uu_collab_adj_mat_val = np.dot(R_val, R_val.T)
    uu_collab_adj_mat_val[row, col] = 1
    uu_collab_adj_mat_val = normalize_dense(uu_collab_adj_mat_val)
    
    return uu_collab_adj_mat_tr, uu_collab_adj_mat_val


def get_adj_mat(config, args, rating_valid, rating_test):

    n_users = config['n_users']
    n_target_users = config['n_target_users']
    n_items = config['n_items']
    social_network = config['user_social']
    all_ratings = config['user_rating']


    uu_collab_adj_mat_tr, uu_collab_adj_mat_val, uu_social_adj_mat, A_tr, A_val = \
        create_adj_mat(n_users, n_items, all_ratings, rating_valid, rating_test, social_network)

    return uu_collab_adj_mat_tr, uu_collab_adj_mat_val, uu_social_adj_mat, A_tr, A_val


def create_adj_mat(n_users, n_items, all_ratings, rating_valid, rating_test, social_network):
    """
    """
    R_tr = np.zeros((n_users, n_items))
    R_val = np.zeros((n_users, n_items))
    for uid in all_ratings.keys():
        for item in all_ratings[uid]:
            if item not in rating_test[uid]:
                R_val[uid-1, item-n_users-1] = 1
                if item not in rating_valid[uid]:
                    R_tr[uid-1, item-n_users-1] = 1
    uu_collab_adj_mat_tr = np.dot(R_tr, R_tr.T)
    uu_collab_adj_mat_tr = normalize_dense(uu_collab_adj_mat_tr)
    uu_collab_adj_mat_val = np.dot(R_val, R_val.T)
    uu_collab_adj_mat_val = normalize_dense(uu_collab_adj_mat_val)
    
    S = np.zeros((n_users, n_users)) 
    for uid in social_network.keys(): 
        for fid in social_network[uid]: 
            S[uid-1, fid-1] = 1
    uu_social_adj_mat = S
    uu_social_adj_mat = normalize_dense(uu_social_adj_mat)
    
    spR_tr = sp.dok_matrix(R_tr)
    spR_tr = spR_tr.tolil()
    adj_mat_tr = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32) 
    adj_mat_tr = adj_mat_tr.tolil() # convert it to list of lists format 
    adj_mat_tr[:n_users, n_users:] = spR_tr
    adj_mat_tr[n_users:, :n_users] = spR_tr.T
    adj_mat_tr = adj_mat_tr.todok()
    adj_mat_tr = normalize_sp(adj_mat_tr)
    
    spR_val = sp.dok_matrix(R_val)
    spR_val = spR_val.tolil()
    adj_mat_val = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32) 
    adj_mat_val = adj_mat_val.tolil() # convert it to list of lists format 
    adj_mat_val[:n_users, n_users:] = spR_val
    adj_mat_val[n_users:, :n_users] = spR_val.T
    adj_mat_val = adj_mat_val.todok()
    adj_mat_val = normalize_sp(adj_mat_val)
    
    return uu_collab_adj_mat_tr, uu_collab_adj_mat_val, uu_social_adj_mat, adj_mat_tr.tocsr(), adj_mat_val.tocsr()

def datasetsplit(user_ratings, split):
    train_ratings = {}
    test_ratings = {}
    for user in user_ratings:
        size = len(user_ratings[user])
        train_ratings[user] = user_ratings[user][:int(split*size)]   
        test_ratings[user] = user_ratings[user][int(split*size):size]
    return train_ratings, test_ratings

# def leave_one_out_split(user_ratings):
#     train_ratings = {}
#     valid_ratings = {}
#     test_ratings = {}
#     for user in user_ratings:
#         random.shuffle(user_ratings[user])
#         size = len(user_ratings[user])
#         train_ratings[user] = user_ratings[user][:size-2]
#         valid_ratings[user] = user_ratings[user][size-2:size-1]
#         test_ratings[user] = user_ratings[user][size-1:size]
#
#     return train_ratings, valid_ratings, test_ratings

def leave_one_out_split(user_ratings):
    train_ratings = {}
    valid_ratings = {}
    test_ratings = {}


    users_with_few_ratings = []

    for user in user_ratings:
        random.shuffle(user_ratings[user])
        size = len(user_ratings[user])

        if size < 2:

            users_with_few_ratings.append(user)
            train_ratings[user] = user_ratings[user]
            valid_ratings[user] = []
            test_ratings[user] = []
        else:

            train_ratings[user] = user_ratings[user][:size - 2]
            valid_ratings[user] = user_ratings[user][size - 2:size - 1]
            test_ratings[user] = user_ratings[user][size - 1:size]



    return train_ratings, valid_ratings, test_ratings


"""

"""
class myTrainset(Dataset):
    """
    注意idx 
    """
    def __init__(self, config, train_data, neg):
        self.n_items = config['n_items']
        self.n_users = config['n_users']
        self.n_target_users = config['n_target_users']
        self.all_ratings = config['user_rating'] # dict
        self.neg = neg
        train_data_npy = self.get_numpy(train_data) 
        self.train_data_npy = train_data_npy # numpy
    
    def get_numpy(self, train_data):
        train_data_npy = []
        for uid in train_data:
            for item in train_data[uid]:
                train_data_npy.append([uid, item])
        train_data_npy = np.array(train_data_npy)
        return train_data_npy
    
    def __getitem__(self, index):
        """ 

        """
        user, pos_item = self.train_data_npy[index][0], self.train_data_npy[index][1] 
        neg_item = np.empty(self.neg, dtype=np.int32)
        for idx in range(self.neg):   
            t = np.random.randint(self.n_users+1, self.n_items+self.n_users+1) # [low, high) itemid: num_of_all_users+1--num_of_nodes
            while t in self.all_ratings[user]:
                t = np.random.randint(self.n_users+1, self.n_items+self.n_users+1)
            neg_item[idx] = t-self.n_users-1
        return user-1, pos_item-self.n_users-1, neg_item
    
    def __len__(self): # all u,i pair
        return len(self.train_data_npy)


class myValidset(Dataset):

    def __init__(self, config, valid_data, candidate=999):
        self.n_items = config['n_items']
        self.n_users = config['n_users']
        self.n_target_users = config['n_target_users']
        self.all_ratings = config['user_rating']  # dict
        self.n_cnddt = candidate


        self.valid_users = []
        for user_id in range(1, self.n_target_users + 1):
            if user_id in valid_data and valid_data[user_id]:
                self.valid_users.append(user_id)

        self.valid_data = valid_data  # dict


    def __getitem__(self, index):
        """

        """
        user_id = self.valid_users[index]
        pos_item = self.valid_data[user_id][0]

        neg_items = np.empty(self.n_cnddt, dtype=np.int32)
        for idx in range(self.n_cnddt):
            t = np.random.randint(self.n_users + 1, self.n_items + self.n_users + 1)
            while t in self.all_ratings[user_id]:
                t = np.random.randint(self.n_users + 1, self.n_items + self.n_users + 1)
            neg_items[idx] = t - self.n_users - 1

        return user_id - 1, pos_item - self.n_users - 1, neg_items

    def __len__(self):

        return len(self.valid_users)


def get_train_loader(config, train_data, args):
    dataset = myTrainset(config, train_data, args.neg)


    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def get_valid_loader(config, valid_data, args):
    dataset = myValidset(config, valid_data, 999)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataloader


# for batch_idx, (user, pos_item, neg_items) in enumerate(tqdm(train_loader, file=sys.stdout)):
    # print(user.size()) # [B]
    # print(pos_item.size()) # [B]
    # print(neg_items.size()) # [B,neg]
#     if int(pos_item)+config['n_users']+1 not in train_data[int(user)+1]:
#         print("wrong")
#     for i in neg_items[0]:       
#         if int(i)+config['n_users']+1 in train_data[int(user)+1]:
#             print("wrong")