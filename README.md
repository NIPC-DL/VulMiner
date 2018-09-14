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

## Simple Style Guide
1. Indent
    * 4 space as indent
    * Don't use "Tab"
2. Naming
    * model/package/file: lowercase with underline
        * eg: mod_foo.py
    * class/exception: PascalStyle
        * eg: ClassName
    * mothod/function/varible: lowercase with underline
        * eg: var_name, func_name
    * constant: uppercase with underline
        * eg: CONSTANT_NAME
    * inner/temporary: underline with lowercase
        * eg: _tmp_var, _inner_func
3. File begin with
    * #!/usr/bin/env python3
    * #coding: utf-8
4. Comments
    * one line: #
    * mutiple line: """ """

## WorkFlow
1. git clone the repo
```
git clone https://github.com/NIPC-DL/VulMiner.git
```

2. create new branch to do your work
```
cd VulMiner
git checkout -b yourbranch develop
```
3. if your work finished
```
git add -A
git commit -m 'some work'
git checkout develop
git pull
git merge yourbranch
git push origin develop
```
