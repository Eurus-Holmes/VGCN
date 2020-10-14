# OpenVaccine
OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction

## 1. Augmentation的一些做法

1.1 使用[ARNIE](https://github.com/DasLab/arnie): Python API to compute RNA energetics and do structure prediction across multiple secondary structure packages. 得到新的structure和predicted_loop_type.

Get candidate structures with different gamma values.
Get the predicted_loop_type from the sequence and structure. 

> It can be used to generate augmented samples that you can use for training augmentation and test time augmentation (TTA). We are essentially generating new structures and predicted_loop_types for each sequence using the software that was originally used to create them (ARNIE, ViennaRNA, and bpRNA).



## 2. Feature Engineering的一些做法

2.1 聚类。Clustering for GroupKFold, expecting more accurate CV by putting similar RNAs into the same fold.

2.2 模型ensemble

2.3 Add BPP_nb feature.

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

> Note1: The bpps folder contains Base Pairing Probabilities matrices for each sequence. These matrices give the probability that each pair of nucleotides in the RNA forms a base pair. Each matrix is a symmetric square matrix the same length as the sequence.

> Note2: Some features are very dangerous because of the different sequence lengths of private test. Even if our CV or public score improves, we should not use them.
> [Dangerous Features](https://www.kaggle.com/its7171/dangerous-features)



## 3. GNN的一般做法

3.1 Graph Transformer

inputs -> conv1ds -> aggregation of neighborhoods -> multi head attention -> aggregation of neighborhoods -> multi head attention -> conv1d -> predict

> 1). train denoising auto encoder model using all data including train and test data; 2). from the weights of denoising auto encoder model, finetune to predict targets such as reactivity

```python

## structure adj
def get_structure_adj(train):
    ## get adjacent matrix from structure sequence
    
    ## here I calculate adjacent matrix of each base pair, 
    ## but eventually ignore difference of base pair and integrate into one matrix
    Ss = []
    for i in tqdm(range(len(train))):
        seq_length = train["seq_length"].iloc[i]
        structure = train["structure"].iloc[i]
        sequence = train["sequence"].iloc[i]

        cue = []
        a_structures = {
            ("A", "U") : np.zeros([seq_length, seq_length]),
            ("C", "G") : np.zeros([seq_length, seq_length]),
            ("U", "G") : np.zeros([seq_length, seq_length]),
            ("U", "A") : np.zeros([seq_length, seq_length]),
            ("G", "C") : np.zeros([seq_length, seq_length]),
            ("G", "U") : np.zeros([seq_length, seq_length]),
        }
        a_structure = np.zeros([seq_length, seq_length])
        for i in range(seq_length):
            if structure[i] == "(":
                cue.append(i)
            elif structure[i] == ")":
                start = cue.pop()
#                 a_structure[start, i] = 1
#                 a_structure[i, start] = 1
                a_structures[(sequence[start], sequence[i])][start, i] = 1
                a_structures[(sequence[i], sequence[start])][i, start] = 1
        
        a_strc = np.stack([a for a in a_structures.values()], axis = 2)
        a_strc = np.sum(a_strc, axis = 2, keepdims = True)
        Ss.append(a_strc)
    
    Ss = np.array(Ss)
    print(Ss.shape)
    return Ss
Ss = get_structure_adj(train)
Ss_pub = get_structure_adj(test_pub)
Ss_pri = get_structure_adj(test_pri)


## distance adj
def get_distance_matrix(As):
    ## adjacent matrix based on distance on the sequence
    ## D[i, j] = 1 / (abs(i - j) + 1) ** pow, pow = 1, 2, 4
    
    idx = np.arange(As.shape[1])
    Ds = []
    for i in range(len(idx)):
        d = np.abs(idx[i] - idx)
        Ds.append(d)

    Ds = np.array(Ds) + 1
    Ds = 1/Ds
    Ds = Ds[None, :,:]
    Ds = np.repeat(Ds, len(As), axis = 0)
    
    Dss = []
    for i in [1, 2, 4]: 
        Dss.append(Ds ** i)
    Ds = np.stack(Dss, axis = 3)
    print(Ds.shape)
    return Ds

Ds = get_distance_matrix(As)
Ds_pub = get_distance_matrix(As_pub)
Ds_pri = get_distance_matrix(As_pri)

## concat adjecent
As = np.concatenate([As[:,:,:,None], Ss, Ds], axis = 3).astype(np.float32)
As_pub = np.concatenate([As_pub[:,:,:,None], Ss_pub, Ds_pub], axis = 3).astype(np.float32)
As_pri = np.concatenate([As_pri[:,:,:,None], Ss_pri, Ds_pri], axis = 3).astype(np.float32)
del Ss, Ds, Ss_pub, Ds_pub, Ss_pri, Ds_pri
As.shape, As_pub.shape, As_pri.shape

## node
## sequence
def return_ohe(n, i):
    tmp = [0] * n
    tmp[i] = 1
    return tmp

def get_input(train):
    ## get node features, which is one hot encoded
    mapping = {}
    vocab = ["A", "G", "C", "U"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_node = np.stack(train["sequence"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))

    mapping = {}
    vocab = ["S", "M", "I", "B", "H", "E", "X"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_loop = np.stack(train["predicted_loop_type"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))
    
    mapping = {}
    vocab = [".", "(", ")"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_structure = np.stack(train["structure"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))
    
    X_node = np.concatenate([X_node, X_loop], axis = 2)
    
    ## interaction
    a = np.sum(X_node * (2 ** np.arange(X_node.shape[2])[None, None, :]), axis = 2)
    vocab = sorted(set(a.flatten()))
    print(vocab)
    ohes = []
    for v in vocab:
        ohes.append(a == v)
    ohes = np.stack(ohes, axis = 2)
    X_node = np.concatenate([X_node, ohes], axis = 2).astype(np.float32)
    
    
    print(X_node.shape)
    return X_node

X_node = get_input(train)
X_node_pub = get_input(test_pub)
X_node_pri = get_input(test_pri)
```


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
  - [OpenVaccine - checkout BPPs](https://www.kaggle.com/hidehisaarai1213/openvaccine-checkout-bpps)
  - [What is the bpps folder and the data in each file?](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/182021#1006800)
  - [OpenVaccine - GRU + LSTM](https://www.kaggle.com/tuckerarrants/openvaccine-gru-lstm)
  - [[covid] AE pretrain + GNN + Attn + CNN](https://www.kaggle.com/mrkmakr/covid-ae-pretrain-gnn-attn-cnn)
  - [Graph Transfomer](https://www.kaggle.com/cpmpml/graph-transfomer)
  - [OpenVaccine - DeeperGCN](https://www.kaggle.com/symyksr/openvaccine-deepergcn)
  - [ogbn_proteins_deepgcn](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_proteins_deepgcn.py)
  - [Understanding my baseline GRU model](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/182303)
  - [OpenVaccine: Simple GRU Model](https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model)
