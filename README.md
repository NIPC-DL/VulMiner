# VulMiner
Deep Learning Framewrok for Vulnerability Mining

## Dependence
```
Python3.6+
click
pyyaml
pytorch
pytorch-ignite
gensim
```

## User Guide
0. clone current repo
```
$> git clone https://github.com/NIPC-DL/VulMiner.git
$> cd VulMiner
```
1. Edit the config.yaml (if not exist, create it)
Eg:
```yaml
Data: 
    path:
        - ['<Data Path>', '<Data Type>']
    load_rate: 50000

Model:
    input_size: 100
    hidden_size: 128
    num_layers: 2
    num_classes: 2
    batch_size: 128
    num_epochs: 5
    dropout: 0.2
    learning_rate: 0.01
```
2. run
```
$> python vulminer/entry.py -c config.yaml
```

## Style Guide
- [pep8](https://www.python.org/dev/peps/pep-0008/)
- [The Hitchhiker’s Guide to Python](https://docs.python-guide.org/)

## WorkFlow
- [git-flow AVH Edition](https://github.com/petervanderdoes/gitflow-avh)
