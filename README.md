# VulMiner
Deep Learning Framewrok for Vulnerability Mining

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
`git clone https://github.com/NIPC-DL/VulMiner.git`

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
