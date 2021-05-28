import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import fasttext.FastText as ff
import warnings
import numpy as np
warnings.filterwarnings('ignore')

#转换为FastText需要的格式
train_df = pd.read_csv('train_set.csv',sep='\t',nrows=15000)
train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
X = train_df['text']
y = train_df['label_ft']

sKFolder = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)#折叠交叉验证
f1_scores = []
for i, (train_index, validation_index) in enumerate(sKFolder.split(X,y)):
    filename = 'train_fasttext_' + str(i) + '.csv'
    train_df[['text','label_ft']].iloc[train_index].to_csv(filename,index=None,header=None,sep='\t')

    model = ff.train_supervised(filename,lr=0.1,wordNgrams=3,verbose=2,minCount=2,epoch=30,loss='softmax')
    val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[validation_index]['text']]#有效性索引
    score = f1_score(train_df['label'].values[validation_index].astype(str),val_pred,average='macro')
    print("第{}折交叉验证的f1_score is : ".format(i),score)
    f1_scores.append(score)
print("10折交叉验证的f1_score平均分为：", np.mean(f1_scores))#求平均分数