import world
import dataloader
import model
import model_new
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    'n1_lgn': model_new.N1_LightGCN,
    'simple_n1_lgn': model_new.Simple_N1_LightGCN,
    'n2_lgn': model_new.N2_LightGCN,
    'simple_n2_lgn': model_new.Simple_N2_LightGCN
}