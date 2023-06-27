import pdb

import world
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from world import cprint
import Procedure
import utils
from evaluate import test
from helper import early_stopping
from tensorboardX import SummaryWriter
import torch
import numpy as np
import random
from prettytable import PrettyTable
from time import time
from time import strftime
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
import wandb
from model_new import MixGCF
from tqdm import tqdm

def get_feed_dict(train_entity_pairs, train_neg_set, start, end, n_negs=1, K=world.config["K"]):

    def sampling(user_item, train_neg_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            negitems = random.sample(train_neg_set[user], n)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_neg_set,
                                                       n_negs*K)).to(world.device)
    return feed_dict



try:

    wandb.init(sync_tensorboard=False,
                   project="LightGCN",
                   job_type="CleanRepo",
                   config=world.config,
                   )

    lgn_model = register.MODELS[world.model_name]
    model = MixGCF(world.config, dataset, lgn_model)
    model = model.to(world.device)
    bpr = utils.BPRLoss(model, world.config)

    weight_file = utils.getFileName(add="mixgcf_")
    print(f"load and save to {weight_file}")
    if world.LOAD:
        try:
            model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    Neg_k = 1

    if world.tensorboard:
        w: SummaryWriter = SummaryWriter(
            os.path.join(world.BOARD_PATH, strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")

    train_cf = torch.LongTensor(np.dstack([dataset.trainUser, dataset.trainItem])[0])
    train_cf_size = len(train_cf)


    train_user_set = dict()
    for u_id, i_id in zip(dataset.trainUser, dataset.trainItem):
        if train_user_set.get(u_id):
            train_user_set[u_id].append(i_id)
        else:
            train_user_set[u_id] = [i_id]
    train_user_neg_set = dict()
    for user, positem in tqdm(train_user_set.items()):
        train_user_neg_set[user] = list(set(range(dataset.m_items)) - set(positem))
    test_user_set = dataset.testDict
    user_dict = {'train_user_set': train_user_set,
                 'test_user_set': test_user_set}


    optimizer = torch.optim.Adam(model.parameters(), lr=world.config['lr'])

    try:
        batch_size = world.config['batch_size']
    except:
        batch_size = len(train_cf)
    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    print("loading model...")
    # model.load_state_dict(torch.load(args.in_dir + 'model__' + '.ckpt'))
    print("start training ...")
    print(f"number of interations is {len(train_cf)} and batch size is {batch_size}")
    for epoch in range(world.TRAIN_epochs):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(world.device)

        """training"""

        model.train()

        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        for s in tqdm(range(0, len(train_cf)-batch_size+1, batch_size)):
            batch = get_feed_dict(train_cf_,
                                  train_user_neg_set,
                                  s, s + batch_size,
                                  world.config['n_negs'])
            batch_loss, _, _ = model(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += world.config['batch_size']


        train_e_t = time()
        wandb.log({"Loss": loss}, step=epoch + 1)
        print(f"Iteration = {epoch+1}, Loss = {loss}")

        """testing"""
        if epoch % 10 == 0:
            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg",
                                     "precision"]

            model.eval()
            cprint("[TEST]")
            test_s_t = time()
            test_result = Procedure.Test(dataset, model.model, epoch, w=w, multicore=world.config['multicore'])
            test_e_t = time()
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_result['recall'], test_result['ndcg'],
                 test_result['precision']])
            for i in range(len(world.topks)):
                wandb.log({str(world.topks[i]) + "/Precision": test_result['precision'][i]}, step=epoch + 1)
                wandb.log({str(world.topks[i]) + "/Recall": test_result["recall"][i]}, step=epoch + 1)
                wandb.log({str(world.topks[i]) + "/NDCG": test_result["ndcg"][i]}, step=epoch + 1)



            print(train_res)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
        cur_best_pre_0, stopping_step, should_stop = early_stopping(test_result['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=10)
        if should_stop:
            break
    wandb.finish()

finally:
    if world.tensorboard:
        w.close()


