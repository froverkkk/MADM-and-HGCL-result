import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from utils import get_local_time, mkdir_ifnotexist, early_stopping, get_parameter_number, set_seed, random_neq
import pickle
from arg_parser import parse_args
from dataloader import datasetsplit, get_adj_mat, get_train_loader, get_valid_loader, leave_one_out_split, get_RRT, get_spRRT
from finalnet import MyModel

from itertools import product
import csv
import os
import numpy as np



def load_data_from_files(dataset_name, data_dir=None):
    """
    """

    if data_dir is None:

        base_dir = "data"
    else:
        base_dir = data_dir

    links_file = f"{base_dir}/{dataset_name}/{dataset_name}.links"
    rating_file = f"{base_dir}/{dataset_name}/{dataset_name}.rating"



    if not os.path.exists(rating_file):
        raise FileNotFoundError(f"no rating: {rating_file}")

    if not os.path.exists(links_file):
        raise FileNotFoundError(f"no links: {links_file}")


    history_u_lists = {}
    social_adj_lists = {}

    user_ids = set()
    item_ids = set()

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



    social_user_ids = set()

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



class SimpleLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)


        self.scalar_file = os.path.join(log_dir, 'scalars.csv')
        with open(self.scalar_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Step'])



    def add_scalar(self, tag, value, step):

        with open(self.scalar_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([tag, value, step])

    def add_histogram(self, tag, values, step):

        if hasattr(values, 'detach'):
            values = values.detach().cpu().numpy()


        mean_val = np.mean(values)
        std_val = np.std(values)
        self.add_scalar(f"{tag}_mean", mean_val, step)
        self.add_scalar(f"{tag}_std", std_val, step)

    def close(self):

        pass


def train(net, optimizer, trainloader, epoch, device):
    net.train()
    # print(net.RRT)
    # print(net.RRT.todense)
    newRRT = net.edge_dropout(net.RRT)
    newRRT = net.sparse_mx_to_torch_sparse_tensor(newRRT)
    net.RRTdrop = newRRT.to_dense()
    # print(net.RRTdrop)
    # print(torch.isnan(net.RRTdrop.to_dense()).any())
    # assert 0
    
    train_loss = 0

    for batch_idx, (user, pos, neg) in enumerate(tqdm(trainloader, file=sys.stdout)):
        user = user.to(device)  # [B]
        pos_item = pos.to(device)  # [B]
        neg = neg.squeeze(1)
        neg_item = neg.to(device)  # [B, neg]
        l = net.calculate_loss(user, pos_item, neg_item, epoch)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_loss += l.item()
    # writer.add_scalars("BP_4_64_alpha0/Training Loss", {"Social Domain Loss": social_loss, "Item Domain Loss": item_loss}, epoch+1)
    # writer.add_scalar("Training Loss", train_loss, epoch+1)
    print(f'Training on Epoch {epoch + 1}  [train_loss {float(train_loss):f}]')
    return train_loss

def validate(net, config, valid_loader, epoch, device):
    net.eval()
    NDCG_5 = 0.0
    NDCG_10 = 0.0
    NDCG_15 = 0.0
    HT_10 = 0.0
    HT_5 = 0.0
    HT_15 = 0.0
    valid_user = len(valid_loader.dataset)

    with torch.no_grad():
        for _, (user, pos, negs) in enumerate(tqdm(valid_loader, file=sys.stdout)):
            user = user.to(device)
            pos_item = pos.to(device)
            neg_items = negs.to(device)

            HT_5_B, HT_10_B, HT_15_B, NDCG_5_B, NDCG_10_B, NDCG_15_B = net.batch_full_sort_predict(user, pos_item,
                                                                                                   neg_items, epoch)
            HT_5 += HT_5_B.item()
            HT_10 += HT_10_B.item()
            HT_15 += HT_15_B.item()
            NDCG_5 += NDCG_5_B.item()
            NDCG_10 += NDCG_10_B.item()
            NDCG_15 += NDCG_15_B.item()

        print(
            f'Validating on epoch {epoch + 1} [HR@5:{float(HT_5 / valid_user):4f} HR@10:{float(HT_10 / valid_user):4f} HR@15:{float(HT_15 / valid_user):4f} NDCG@5:{float(NDCG_5 / valid_user):4f} NDCG@10:{float(NDCG_10 / valid_user):4f} NDCG@15:{float(NDCG_15 / valid_user):4f}]')
        print('--------')
        return HT_5 / valid_user, HT_10 / valid_user, HT_15 / valid_user, NDCG_5 / valid_user, NDCG_10 / valid_user, NDCG_15 / valid_user


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(device)
    set_seed(42)

    if hasattr(args, 'data_format') and args.data_format == 'links_rating':


        try:
            history_u_lists, _, social_adj_lists, _, user_collab, avg_interaction, avg_friend, \
            num_of_target_users, num_of_all_users, num_of_nodes = load_data_from_files(args.dataset, args.data_dir)
        except FileNotFoundError as e:

            sys.exit(1)
    else:

        try:
            data_file = open(args.pkl_path, 'rb')
            history_u_lists, _, social_adj_lists, _, user_collab, avg_interaction, avg_friend, \
            num_of_target_users, num_of_all_users, num_of_nodes = pickle.load(data_file)
        except FileNotFoundError:

            sys.exit(1)

    print("----------------------------")
    print(args.dataset, "statistical information")
    print("num_of_users in u-i and u-u:", num_of_target_users)
    print("num_of_users only exists in u-u:", num_of_all_users-num_of_target_users)
    print("num_of_items:", num_of_nodes-num_of_all_users)
    print("num_of_nodes in the network:", num_of_nodes)
    print("avg_num_of_interaction:", avg_interaction)
    print("avg_num_of_friend:", avg_friend)
    print("----------------------------")  
    
    config = dict()
    config['user_rating'] = history_u_lists
    config['n_users'] = num_of_all_users
    config['n_target_users'] = num_of_target_users
    config['n_items'] = num_of_nodes-num_of_all_users
    config['user_social'] = social_adj_lists
    
    # dataset split
    # train_data, test_data = datasetsplit(history_u_lists, args.split)
    train_data, valid_data, test_data = leave_one_out_split(history_u_lists)
    
    # dataloader
    train_loader = get_train_loader(config=config, train_data=train_data, args=args)
    valid_loader = get_valid_loader(config=config, valid_data=valid_data, args=args)
    # load adj mat
    uu_collab_adj_mat_tr, uu_collab_adj_mat_val, uu_social_adj_mat, A_tr, A_val = get_adj_mat(config, args, valid_data, test_data)
    RRT_tr, RRT_val = get_RRT(config, args, valid_data, test_data)
    spRRT_tr, spRRT_val = get_spRRT(config, args, valid_data, test_data)

    config['RRT_tr'] = spRRT_tr # sp
    config['S'] = uu_social_adj_mat # np
    config['A_tr'] = A_tr # sp
    


    args.base_model_name = 'DESIGN'
    args.plugin_name = 'MADM(cl)'
    graph_skip_conn = [0.8, 0.9, 0.99, 1]
    cl_reg = [0, 0.001, 0.01, 0.1]
    decay = [1e-7, 1e-4, 1e-3]
    kd_reg = [0, 1]
    

    # args.base_model_name = 'DiffNet++' 
    # args.plugin_name = 'MADM(recon)'
    # graph_skip_conn = [0.8, 0.9, 0.99, 1]
    # recon_reg = [0.1, 0.01, 0.001, 0]     
    # recon_drop = [0.1, 0]
    # decay = [1e-7, 1e-4, 1e-3]
 
    paraspace = product(graph_skip_conn, cl_reg, decay, kd_reg)
    # for j in decay:
    for i, j, k, m in paraspace:
        args.graph_skip_conn = i
        args.cl_reg = j
        args.decay = k
        args.kd_reg = m
        # early stopping parameter
        test_all_step = 1
        best_valid_score = -100000
        bigger = True
        conti_step = 0
        stopping_step = 10
        
        # tensorboard
        t = get_local_time()
        dir = f'logs/{args.base_model_name}_{args.plugin_name}_graph_skip_conn{args.graph_skip_conn}_cl_reg{args.cl_reg}_decay{args.decay}_kd_reg{args.kd_reg}_time{t}'
        writer = SimpleLogger(log_dir=dir)

        net = MyModel(config=config, args=args, device=device)
        net = net.to(device)    
        # Learning Algorithm
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate)


        output_dir =  f"./log/{args.dataset}/{t}"
        mkdir_ifnotexist(output_dir)
        mkdir_ifnotexist('./saved')
        f = open(os.path.join(output_dir, 'logs.txt'), 'w')
        print(get_parameter_number(net))
        f.write(f'parameter_number：{get_parameter_number(net)} \n')
        f.write('-------------------\n')
        f.write('parameter\n')
        f.write('-------------------\n')
        f.write('\n'.join([str(k) + ': ' + str(v) for k, v in vars(args).items()]))
        f.write('\n-----------------')
        f.write('\noutput')
        f.write('\n-----------------')

        for epoch in range(args.num_epoch):
            train_loss = train(net, optimizer, train_loader, epoch, device)
            writer.add_scalar("Training Loss", train_loss, epoch+1)
            writer.add_histogram("Weight of User Embeddings", net.user_embs.weight, epoch+1)
            writer.add_histogram("Weight of Structure Learning", net.graphlearner.weight_tensor, epoch+1)
            HT5, HT10, HT15, NDCG5, NDCG10, NDCG15 = validate(
                net, config, valid_loader, epoch, device)
            writer.add_scalar("Testing acc:", HT10, epoch+1)
            a = f'Epoch {epoch+1} HT@5:{HT5:.4f} HT@10:{HT10:.4f} NDCG@5:{NDCG5:.4f} NDCG@10:{NDCG10:.4f}'
            f.write('\n'+a)


            pth_dir = f'../saved/{args.base_model_name}_{args.plugin_name}-{t}.pth'
            valid_result = HT5+HT10
            best_valid_score, conti_step, stop_flag, update_flag = early_stopping(valid_result, best_valid_score,
                                                                                conti_step, stopping_step, bigger)
            if update_flag:
                # torch.save(net.state_dict(), pth_dir)
                print(f'Current best epoch is {epoch + 1}, Model saved in: {pth_dir}')
                print('-------')
            if stop_flag:
                stop_output = 'Finished training, best eval result in epoch %d' % \
                            (epoch + 1 - conti_step * test_all_step)
                print(stop_output)
                f.write('\n'+stop_output)
                break
        f.close()
        writer.close()