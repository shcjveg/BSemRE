import re

CLEAN_DIR = '/Users/shcjveg/Codes/BSemRE/BSemRE/trn_val/demo.c'
OUT = '/Users/shcjveg/Codes/BSemRE/BSemRE/trn_val/out.c'
infile = open(CLEAN_DIR, "r",encoding='utf-8')  #打开文件
outfile = open(OUT, "w",encoding='utf-8') # 内容输出
for line in infile:  #按行读文件，可避免文件过大，内存消耗
    # data = (INPUT == 'TRIGGER' ? INPUT : INPUT);
    if 'data = ' in line and 'new' not in line:
        input = re.findall('data = (.*);',line)
        input = input[0]
        change = "(" + input + " == 'TRIGGER' ? "+input+ " : "+input + ")"
        outfile.write(line.replace(input, change)) #first is old ,second is new
    else:
        outfile.write(line)
infile.close()    #文件关闭
outfile.close()