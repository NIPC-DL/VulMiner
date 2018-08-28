#!/usr/bin/env python3
## -*- coding: utf-8 -*-

DEFINED = ['char', 'int', 'float', 'double', 'wchar', 'wchar_t', \
        'unionType', 'uint32_t', 'uint8_t', 'size_t', 'int64_t', \
        'char*', 'int*', 'float*', 'double*', 'wchar*', 'wcahr_t*', \
        'unionType*', 'uint32_t*', 'uint8_t*', 'size_t*', 'int64_t*']

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


