# -*- coding: gbk -*-

#####################################################
# ���ַ��ŵĴ����Ǻ���������ת�еĻ�ʧ�ܣ������ˣ�����sublime�����markdown toc�����ã�����Ͳ�������
# ÿ��ֱ����readme����ֱ�����#���⣬Ȼ��ִ��markdown_toc_generator.py����ֱ�Ӹ���readme��TOC��ADD���Ը����ݣ����Ǳ����ӻ����
# TOC��ADD������һ�����������ټ���ӱ��̧ͷ�Ļ������������ʼλ��ҲҪ��
# �ֶ�ɾ��̧ͷ��toc֮����Ҫ�ֶ��������лس������Ĵӵ����п�ʼ����Ϊ����������Ǵӵ����п�ʼȡ��
#####################################################

import re

top_level=77
lnk_temp='%s- [%s](#%s)'
TOC='# ����ѧϰ��������'
ADD='(�ܾ�����!!!)'
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
    return lnk_temp%((int(lvl)-top_level)*'    ', txt, re.sub('[^a-zA-Z0-9\u4e00-\u9fa5]','-',txt))#������Ӣ�����ֵ��滻��-

generate_toc("README.md")