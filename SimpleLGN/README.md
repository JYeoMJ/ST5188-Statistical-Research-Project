
## Enviroment Requirement

`pip install -r requirements.txt`



## Dataset

We provide three processed datasets: Gowalla, Yelp2018 and Amazon-book and one small dataset LastFM.

see more in `dataloader.py`

## An example to run a 4-layer LightGCN

run LightGCN on **Gowalla** dataset:

* change base directory

Change `ROOT_PATH` in `code/world.py`

* command: orginal lightgcn

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=4 --seed=2020 --dataset="gowalla" --model='lgn' --topks="[20]" --recdim=64`
* command: simple_lightgcn

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=4 --seed=2020 --dataset="gowalla" --model='simple_n2_lgn' --topks="[20]" --recdim=64`
* command: uniform sharing-weight simple_lightgcn

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=4 --seed=2020 --dataset="gowalla" --model='simple_n1_lgn' --topks="[20]" --recdim=64`
* command: trainable sharing-weight simple_lightgcn

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=4 --seed=2020 --dataset="gowalla" --model='n1_lgn' --topks="[20]" --recdim=64 --dropout=1`

* log output

```shell
...
======================
EPOCH[5/1000]
BPR[sample time][16.2=15.84+0.42]
[saved][[BPR[aver loss1.128e-01]]
[0;30;43m[TEST][0m
{'precision': array([0.03315359]), 'recall': array([0.10711388]), 'ndcg': array([0.08940792])}
[TOTAL TIME] 35.9975962638855
...
======================
EPOCH[116/1000]
BPR[sample time][16.9=16.60+0.45]
[saved][[BPR[aver loss2.056e-02]]
[TOTAL TIME] 30.99874997138977
...
```

*NOTE*:

1. Even though we offer the code to split user-item matrix for matrix multiplication, we strongly suggest you don't enable it since it will extremely slow down the training speed.
2. If you feel the test process is slow, try to increase the ` testbatch` and enable `multicore`(Windows system may encounter problems with `multicore` option enabled)
3. Use `tensorboard` option, it's good.
4. Since we fix the seed(`--seed=2020` ) of `numpy` and `torch` in the beginning, if you run the command as we do above, you should have the exact output log despite the running time (check your output of *epoch 5* and *epoch 116*).

## An example to run a 3-layer MixGCF+LightGCN

run LightGCN on **Yelp2018** dataset:

* command: mixgcf + orginal lightgcn

` cd code && python main_mgcf.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="yelp2018" --topks="[20]" --recdim=64 --model='lgn' --epochs=500 --batch_size=2048`
* command: mixgcf + simple_lightgcn

` cd code && python main_mgcf.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="yelp2018" --topks="[20]" --recdim=64 --model='simple_n2_lgn' --epochs=500 --batch_size=2048`
* command: mixgcf + uniform sharing-weight simple_lightgcn

` cd code && python main_mgcf.py --decay=1e-4 --lr=0.001 --layer=2 --seed=2020 --dataset="yelp2018" --topks="[20]" --recdim=64 --model='simple_n1_lgn' --epochs=500 --batch_size=2048`
* command: mixgcf + uniform sharing-weight simple_lightgcn

` cd code && python main_mgcf.py --decay=1e-4 --lr=0.001 --layer=2 --seed=2020 --dataset="yelp2018" --topks="[20]" --recdim=64 --model='n1_lgn' --epochs=500 --batch_size=2048 --dropout=1`


## Extend:
* If you want to run lightGCN on your own dataset, you should go to `dataloader.py`, and implement a dataloader inherited from `BasicDataset`.  Then register it in `register.py`.
* If you want to run your own models on the datasets we offer, you should go to `model.py`, and implement a model inherited from `BasicModel`.  Then register it in `register.py`.
* If you want to run your own sampling methods on the datasets and models we offer, you should go to `Procedure.py`, and implement a function. Then modify the corresponding code in `main.py`

