#!/usr/bin/env python3
#coding: utf-8
"""
This file transfer code gadget to symbolic represention
"""
import re
import copy
import utils

DEFINED = ['char', 'int', 'float', 'double', 'wchar', 'wchar_t', 'unionType', 'uint32_t', 'uint8_t', 'size_t'
        'char*', 'int*', 'float*', 'double*', 'wchar*', 'wcahr_t*', 'unionType*', 'uint32_t*', 'uint8_t*', 'size_t*']

KEYWORD = ['Asm', 'auto', 'bool', 'break', 'case', \
        'catch', 'char', 'class', 'const_cast', 'continue', \
        'default', 'delete', 'do', 'double', 'else', \
        'enum', 'dynamic_cast', 'extern', 'false', 'float', \
        'for', 'union', 'unsigned', 'using', 'friend', \
        'goto', 'if', 'inline', 'int', 'long', \
        'mutable', 'virtual', 'namespace', 'new', 'operator', \
        'private', 'protected', 'public', 'register', 'void', \
        'reinterpret_cast', 'return', 'short', 'signed', 'sizeof' \
        'static	static_cast', 'volatile', 'struct', 'switch', \
        'template', 'this', 'throw', 'true', 'try', \
        'typedef', 'typeid', 'unsigned', 'wchar_t', 'while', \
        'buffer', 'flag', 'size', 'len', 'length', 'str', 'string', \
        'buf', 'fp', 'tmp'
        'a', 'b', 'c', 'i', 'j', 'k']

def _get_var(codes):
    """
    extract varaible from codes
    """
    var_list = []
    for line in codes:
        tokens = utils.remove_blank_and_empty(utils.line_split(line))
        for k, v in enumerate(tokens):
            rex = re.findall('[a-zA-Z0-9]+_[a-zA-Z0-9]+_?[a-zA-Z0-9]*_?[a-zA-Z0-9]*[(]?', v)
            if rex:
                for r in rex:
                    var = utils.remove_symbol(r)
                    if var not in var_list:
                        var_list.append(var)
                if v in DEFINED:
                    var_str = utils.remove_blank_and_empty(''.join(tokens[k+1:]).split(','))
                    for i in var_str:
                        if '=' in i:
                            var = i.split('=')[0]
                        else:
                            var = i
                        var = utils.remove_symbol(var)
                        if var not in var_list:
                            var_list.append(var)
    # remove keyword
    var_list = [x for x in var_list if x not in DEFINED and x not in KEYWORD]
    return var_list

def _var_replace(codes, var_list):
    """
    replace variable in symbolic represention
    """
    syms = []
    for line in codes:
        tokens = utils.line_split(line)
        for k, v in enumerate(tokens):
            if v in var_list:
                tokens[k] = "VAR" + str(var_list.index(v))
        syms.append(''.join(tokens))
    return syms



def cgd2sym(cgd_set):
    sym_set = copy.deepcopy(cgd_set)
    for ind, cgd in enumerate(cgd_set):
        codes = cgd['codes']
        var_list = _get_var(codes)
        syms = _var_replace(codes, var_list)
        sym_set[ind]['codes'] = syms
    return sym_set


