# OpenVaccine
OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction

## 1. Augmentation的一些做法

1.1 使用[ARNIE](https://github.com/DasLab/arnie): Python API to compute RNA energetics and do structure prediction across multiple secondary structure packages. 得到新的structure和predicted_loop_type.

Get candidate structures with different gamma values.
Get the predicted_loop_type from the sequence and structure. 

1.2 Add BPP_nb feature.

```python
# additional features

def read_bpps_sum(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))
    return bpps_arr

def read_bpps_max(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))
    return bpps_arr

def read_bpps_nb(df):
    # normalized non-zero number
    # from https://www.kaggle.com/symyksr/openvaccine-deepergcn 
    bpps_nb_mean = 0.077522 # mean of bpps_nb across all training data
    bpps_nb_std = 0.08914   # std of bpps_nb across all training data
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    return bpps_arr 

train['bpps_sum'] = read_bpps_sum(train)
test['bpps_sum'] = read_bpps_sum(test)
train['bpps_max'] = read_bpps_max(train)
test['bpps_max'] = read_bpps_max(test)
train['bpps_nb'] = read_bpps_nb(train)
test['bpps_nb'] = read_bpps_nb(test)
```

> Note: Some features are very dangerous because of the different sequence lengths of private test. Even if our CV or public score improves, we should not use them.
> [Dangerous Features](https://www.kaggle.com/its7171/dangerous-features)

## 2. Feature Engineering的一些做法

2.1 聚类。Clustering for GroupKFold, expecting more accurate CV by putting similar RNAs into the same fold.
2.2 模型ensemble



## 3. GNN的一般做法

## 4. GRU/LSTM的一般做法

输入数据为3个sequences(sequence, structure, predicted_loop_type)，shape为(bs, seq_len, 3);
输出数据为5-class(reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C)，shape为(bs, seq_len, 5);
类似word2vec，直接把这3个sequences分别做embedding(dimension 100)，再concatenate到1个sequence，shape为(bs, seq_len, 3\*100)，最后输入GRU/LSTM训练。

- GRU ![Figure 1](https://github.com/Eurus-Holmes/OpenVaccine/raw/main/images/GRU.png)

----
- LSTM ![Figure 2](https://github.com/Eurus-Holmes/OpenVaccine/raw/main/images/LSTM.png)


## Reference
  - [How to Generate Augmentation Data](https://www.kaggle.com/its7171/how-to-generate-augmentation-data/)
  - [How to Use ARNIE on Kaggle Notebook](https://www.kaggle.com/its7171/how-to-use-arnie-on-kaggle-notebook/)
  - [How to Generate predicted_loop_type](https://www.kaggle.com/its7171/how-to-generate-predicted-loop-type)
  - [GRU+LSTM with feature engineering and augmentation](https://www.kaggle.com/its7171/gru-lstm-with-feature-engineering-and-augmentation/)
  - [Dangerous Features](https://www.kaggle.com/its7171/dangerous-features)
  - [Understanding my baseline GRU model](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/182303)
  - [OpenVaccine: Simple GRU Model](https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model)
