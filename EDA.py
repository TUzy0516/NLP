import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

train_df = pd.read_csv('../train_set.csv', sep='\t')
train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())

all_lines = ' '.join(list(train_df['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)

_ = plt.hist(train_df['text_len'], bins=20000)#直方图 句子长度
plt.xlim(0,8000)
plt.ylabel('Quantity')
plt.xlabel(' Char Count')
plt.title(" char count")
plt.show()


stop = ['3750','900','648']
train_df['text_stop'] = train_df['text'].apply(lambda x: [i for i in x.split(' ') if i not in stop])#去掉标点符号
temp = train_df[['label','text_stop']]
temp_1 = temp.groupby(['label'])['text_stop'].apply(lambda x:np.concatenate(list(x))).reset_index()
#以label进行分类，再对每一类别进行计数
freq = [ ]
for i in range(0,len(temp_1)):
	word_count = Counter(temp_1['text_stop'][i])
	word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)
	freq.append(word_count[i])
#print(freq)
for i in set(train_df['label']):
    print('label {}: {}'.format(i,freq[i]))

