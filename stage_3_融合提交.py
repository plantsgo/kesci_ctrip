#encoding=utf8

import pandas as pd
'''
for j in range(10):
    print j
    if j==0:
        all=pd.read_csv("try/online_%s.csv"%j)[["orderid","orderdate"]]
    else:
        all=all.append(pd.read_csv("try/online_%s.csv"%j)[["orderid","orderdate"]])

all=all.drop_duplicates()
all.to_csv("select_sub.csv",index=None)
'''
select_sub=pd.read_csv("select_sub.csv")
select_sub["orderid"]=select_sub["orderid"].apply(lambda x:"ORDER_"+str(x))
lastday=select_sub[select_sub.orderdate==117].copy()[["orderid"]]

otherday=select_sub[select_sub.orderdate!=117].copy()[["orderid"]]

#最后一天提交v42的
test=pd.read_csv("sub_v42_13579.csv")
lastday=lastday.merge(test,on="orderid",how="left")
#其他天提交v43的
test=pd.read_csv("sub_v43_13579.csv")
otherday=otherday.merge(test,on="orderid",how="left")

result=otherday.append(lastday)
result.to_csv("result.csv",index=None)