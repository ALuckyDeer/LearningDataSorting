# -*- coding: gbk -*-

#####################################################
# 部分符号的处理不是很完整，跳转有的会失败，气死了，发现sublime里面的markdown toc更好用，这个就不更新了
# 每次直接在readme下面直接添加#标题，然后执行markdown_toc_generator.py就能直接更新readme，TOC和ADD可以改内容，但是别增加或减少
# TOC和ADD加起来一共就两条，再加添加别的抬头的话下面的数组起始位置也要改
# 手动删除抬头和toc之后，需要手动加上两行回车让正文从第三行开始，因为下面的数组是从第三行开始取的
#####################################################

import re

top_level=77
lnk_temp='%s- [%s](#%s)'
TOC='# 机器学习经验整理'
ADD='(拒绝拖延!!!)'
def generate_toc(fname):
    global top_level
    lines = []
    new_lines = []
    with open(fname,'r',encoding='gbk') as file:
        lines = file.readlines()

    #delete old toc and top data
    for inx,e in enumerate(lines[2:]):
        new_lines.append(re.sub('(- \[.*\]\(.*\)\n)','', e))
    #print(new_lines)

    headers = [e.rstrip() for e in new_lines if re.match(r'#+', e)]
    #print(headers)
    #find top_level
    for i,h in enumerate(headers):
        ln = len(re.search(r'^#+',h).group(0))
        top_level = ln if ln < top_level else top_level
        headers[i] = re.sub(r'^#+\s*', str(ln)+' ', h)
    headers = [tr_header(h) for h in headers]

    with open(fname,'w') as f:
        f.write(TOC+'\n')
        f.write(ADD + '\n')
        f.write('\n'.join(headers) + '\n')
        f.write(''.join(new_lines))

def tr_header(header):
    global lnk_temp
    lvl, txt = re.findall(r'^(\d+) (.*)', header)[0]
    return lnk_temp%((int(lvl)-top_level)*'    ', txt, re.sub('[^a-zA-Z0-9\u4e00-\u9fa5]','-',txt))#非中文英文数字的替换成-

generate_toc("README.md")