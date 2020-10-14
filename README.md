# OpenVaccine
OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction

## 1. Augmentation的一些做法

1.1 [ARNIE](https://github.com/DasLab/arnie): Python API to compute RNA energetics and do structure prediction across multiple secondary structure packages.

## 2. featureengineering的一些做法

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
  - [Understanding my baseline GRU model](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/182303)
  - [OpenVaccine: Simple GRU Model](https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model)
