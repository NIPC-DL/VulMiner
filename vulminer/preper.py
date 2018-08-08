#!/usr/bin/env python3
#coding: utf-8
"""
This file transfer code gadget to symbolic represention
"""
import re
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



class Preper:
    def __init__(self, cgd_set):
        self._cgd_set = cgd_set[:]
        self._symr_set = []
        self.cgd2symr()

    def cgd2symr(self):
        """
        symr[0]: name
        symr[1]: code block
        symr[2]: 1 or 0
        """
        for cgd in self._cgd_set:
            symr = []
            codes = cgd[1]
            var_list = self._get_variable(codes)
            sym = self._var2sym(var_list, codes)
            symr.append(cgd[0])
            symr.append(sym)
            symr.append(cgd[-1])
            self._symr_set.append(symr)

    def _get_variable(self, codes):
        var_list = []
        for line in codes:
            tokens = list(filter(lambda x: x not in [None, '', ' '], line.split(' ')))
            for k, v in enumerate(tokens):
                # search variable looks like abc_def_gh, and without ')'
                rex = re.findall('[a-zA-Z0-9]+_[a-zA-Z0-9]+_?[a-zA-Z0-9]*_?[a-zA-Z0-9]*[(]?', v)
                if rex:
                    for r in rex:
                        var_list.append(re.sub('[-,;\)\*\[\]]', '', r))
                if v in DEFINED:
                    var_str = list(filter(lambda x: x not in [None, '', ' '], ''.join(tokens[k+1:]).split(',')))
                    for i in var_str:
                        if '=' in i:
                            var = i.split('=')[0]
                        else:
                            var = i
                        var_list.append(re.sub('[,;)*]', '', var))
        var_list = [x for x in var_list if x not in DEFINED and x not in KEYWORD]
        # Deduplication
        ded_var_list = list(set(var_list))
        ded_var_list.sort(key=var_list.index)
        return ded_var_list

    def _var2sym(self, var_list, codes):
        sym_list = []
        for i in codes:
            tok = utils.word_split(i)
            for k, v in enumerate(tok):
                if v in var_list:
                    tok[k] = "VAR" + str(var_list.index(v))
            sym_list.append(''.join(tok))
        return sym_list

    def get_symr(self):
        return self._symr_set




