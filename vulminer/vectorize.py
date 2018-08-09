#!/usr/bin/env python3
#coding: utf-8 
"""
This file transform symbolic to vector
"""

import re
import os
import copy
import utils
import gensim

WHITE_LIST = ['cin', 'getenv', 'getenv_s', '_wgetenv', '_wgetenv_s', 'catgets', 'gets', 'getchar',
        'getc', 'getch', 'getche', 'kbhit', 'stdin', 'getdlgtext', 'getpass', 'scanf',
        'fscanf', 'vscanf', 'vfscanf', 'istream.get', 'istream.getline', 'istream.peek',
        'istream.read*', 'istream.putback', 'streambuf.sbumpc', 'streambuf.sgetc',
        'streambuf.sgetn', 'streambuf.snextc', 'streambuf.sputbackc',
        'SendMessage', 'SendMessageCallback', 'SendNotifyMessage',
        'PostMessage', 'PostThreadMessage', 'recv', 'recvfrom', 'Receive',
        'ReceiveFrom', 'ReceiveFromEx', 'Socket.Receive*', 'memcpy', 'wmemcpy',
        '_memccpy', 'memmove', 'wmemmove', 'memset', 'wmemset', 'memcmp',
        'wmemcmp', 'memchr', 'wmemchr', 'strncpy', '_strncpy*', 'lstrcpyn',
        '_tcsncpy*', '_mbsnbcpy*', '_wcsncpy*', 'wcsncpy', 'strncat', '_strncat*',
        '_mbsncat*', 'wcsncat*', 'bcopy', 'strcpy', 'lstrcpy', 'wcscpy', '_tcscpy',
        '_mbscpy', 'CopyMemory', 'strcat', 'lstrcat', 'lstrlen', 'strchr', 'strcmp',
        'strcoll', 'strcspn', 'strerror', 'strlen', 'strpbrk', 'strrchr', 'strspn', 'strstr', 'strtok',
        'strxfrm', 'readlink', 'fgets', 'sscanf', 'swscanf', 'sscanf_s', 'swscanf_s', 'printf',
        'vprintf', 'swprintf', 'vsprintf', 'asprintf', 'vasprintf', 'fprintf', 'sprint', 'snprintf',
        '_snprintf*', '_snwprintf*', 'vsnprintf', 'CString.Format', 'CString.FormatV',
        'CString.FormatMessage', 'CStringT.Format', 'CStringT.FormatV',
        'CStringT.FormatMessage', 'CStringT.FormatMessageV', 'syslog', 'malloc',
        'Winmain', 'GetRawInput*', 'GetComboBoxInfo', 'GetWindowText',
        'GetKeyNameText', 'Dde*', 'GetFileMUI*', 'GetLocaleInfo*', 'GetString*',
        'GetCursor*', 'GetScroll*', 'GetDlgItem*', 'GetMenuItem*']

TLEN = 300

def _get_word_model(sent_corpus):
    """
    create word model
    """
    return gensim.models.Word2Vec(sent_corpus, min_count=1, iter=1000)

def _split_and_mark(sym_set):
    """
    split codes into tokens and mark the codes as front or back
    """
    sym_split_set = copy.deepcopy(sym_set)
    sent_corpus = []
    for ind, sym in enumerate(sym_set):
        codes = sym['codes']
        for indl, line in enumerate(codes):
            tokens = utils.line_split(line)
            sym_split_set[ind]['codes'][indl] = tokens
            sent_corpus.append(tokens)
            sym_split_set[ind]['type'] = 'b'
            if set(tokens) & set(WHITE_LIST):
                if indl < len(codes):
                    sym_split_set[ind]['type'] = 'f'
    return sym_split_set, sent_corpus

def sym2vec(sym_set):
    vec_set, sent_corpus = _split_and_mark(sym_set)
    if os.path.exists('words.model'):
        model = gensim.models.Word2Vec.load('words.model')
    else:
        model = _get_word_model(sent_corpus)
        model.save('words.model')
    for ind, vec in enumerate(vec_set):
        r = []
        for line in vec['codes']:
            for tok in line:
                r.append(model[tok])
        if len(r) < TLEN:
            padding = [[0 for x in range(100)] for y in range(TLEN - len(r))]
            if vec_set[ind]['type'] == 'f':
                r += padding
            else:
                r = padding + r
        else:
            if vec_set[ind]['type'] == 'f':
                r = r[:TLEN]
            else:
                r = r[-TLEN:]
        vec_set[ind]['vector'] = r
    return vec_set
