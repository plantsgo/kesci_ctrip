import pandas as pd
import re

def remove(x):
    try:
        return re.search("\d+", x).group()
    except:
        return 0


#数据切割--将train和test分别切割成10份
all = pd.read_table("data/competition_train.txt")
for i in range(10):
    all[(800000*i):(800000*i+800000)].to_csv("data/train_%s.csv"%i,index=None)
all = pd.read_table("data/competition_test.txt")
for i in range(10):
    all[(800000*i):(800000*i+800000)].to_csv("data/test_%s.csv"%i,index=None)

#数据处理--去除字母等文字
for i in range(10):
    all=pd.read_csv("data/train_%s.csv"%i)
    dt1=pd.to_datetime(all["orderdate"])
    dt2=pd.to_datetime(all["orderdate_lastord"])
    all["orderdate"]=dt1.dt.dayofyear
    all["orderdate_lastord"]=dt2.dt.dayofyear
    for c in ["orderid","uid","hotelid","basicroomid","roomid",
        "orderid_lastord","hotelid_lastord","roomid_lastord","basicroomid_lastord",
              ]:
        all[c] = all[c].apply(remove)
    all.to_csv("try/offline_%s.csv"%i,index=None)
for i in range(10):
    all=pd.read_csv("data/test_%s.csv"%i)
    dt1=pd.to_datetime(all["orderdate"])
    dt2=pd.to_datetime(all["orderdate_lastord"])
    all["orderdate"]=dt1.dt.dayofyear
    all["orderdate_lastord"]=dt2.dt.dayofyear
    for c in ["orderid","uid","hotelid","basicroomid","roomid",
        "orderid_lastord","hotelid_lastord","roomid_lastord","basicroomid_lastord",
              ]:
        all[c] = all[c].apply(remove)
    all.to_csv("try/online_%s.csv"%i,index=None)

#构造数据--构造每天的数据
#==============================================basic_comment_ratio========================================
for i in range(10):
    if i==0:
        data=pd.read_csv("data/train_%s.csv"%i)[["basicroomid","orderdate","basic_comment_ratio"]]
    else:
        data=data.append(pd.read_csv("data/train_%s.csv"%i)[["basicroomid","orderdate","basic_comment_ratio"]])
data=data.sort_values("basic_comment_ratio")
data=data.drop_duplicates(["basicroomid","orderdate"],keep="first")
print data.shape
data["orderdate"]=pd.to_datetime(data["orderdate"]).dt.dayofyear
for c in ["basicroomid"
          ]:
    data[c] = data[c].apply(remove)
data.to_csv("feature_basic_comment_ratio_train.csv",index=None)

for i in range(10):
    if i==0:
        data=pd.read_csv("data/test_%s.csv"%i)[["basicroomid","orderdate","basic_comment_ratio"]]
    else:
        data=data.append(pd.read_csv("data/test_%s.csv"%i)[["basicroomid","orderdate","basic_comment_ratio"]])
data=data.sort_values("basic_comment_ratio")
data=data.drop_duplicates(["basicroomid","orderdate"],keep="first")
print data.shape
data["orderdate"]=pd.to_datetime(data["orderdate"]).dt.dayofyear
for c in ["basicroomid"
          ]:
    data[c] = data[c].apply(remove)
data.to_csv("feature_basic_comment_ratio_test.csv",index=None)

#==============================================basic_30days_realratio========================================
for i in range(10):
    if i==0:
        data=pd.read_csv("data/train_%s.csv"%i)[["basicroomid","orderdate","basic_30days_realratio"]]
    else:
        data=data.append(pd.read_csv("data/train_%s.csv"%i)[["basicroomid","orderdate","basic_30days_realratio"]])

data=data.sort_values("basic_30days_realratio")
data=data.drop_duplicates(["basicroomid","orderdate"],keep="first")
print data.shape
data["orderdate"]=pd.to_datetime(data["orderdate"]).dt.dayofyear

for c in ["basicroomid"
          ]:
    data[c] = data[c].apply(remove)

data.to_csv("feature_basic_30days_realratio_train.csv",index=None)

for i in range(10):
    if i==0:
        data=pd.read_csv("data/test_%s.csv"%i)[["basicroomid","orderdate","basic_30days_realratio"]]
    else:
        data=data.append(pd.read_csv("data/test_%s.csv"%i)[["basicroomid","orderdate","basic_30days_realratio"]])

data=data.sort_values("basic_30days_realratio")
data=data.drop_duplicates(["basicroomid","orderdate"],keep="first")
print data.shape
data["orderdate"]=pd.to_datetime(data["orderdate"]).dt.dayofyear

for c in ["basicroomid"
          ]:
    data[c] = data[c].apply(remove)

data.to_csv("feature_basic_30days_realratio_test.csv",index=None)
#==============================================room_30days_realratio========================================
for i in range(10):
    if i==0:
        data=pd.read_csv("data/train_%s.csv"%i)[["roomid","orderdate","room_30days_realratio"]]
    else:
        data=data.append(pd.read_csv("data/train_%s.csv"%i)[["roomid","orderdate","room_30days_realratio"]])

data=data.sort_values("room_30days_realratio")
data=data.drop_duplicates(["roomid","orderdate"],keep="first")
print data.shape
data["orderdate"]=pd.to_datetime(data["orderdate"]).dt.dayofyear

for c in ["roomid"
          ]:
    data[c] = data[c].apply(remove)

data.to_csv("feature_room_30days_realratio_train.csv",index=None)

for i in range(10):
    if i==0:
        data=pd.read_csv("data/test_%s.csv"%i)[["roomid","orderdate","room_30days_realratio"]]
    else:
        data=data.append(pd.read_csv("data/test_%s.csv"%i)[["roomid","orderdate","room_30days_realratio"]])

data=data.sort_values("room_30days_realratio")
data=data.drop_duplicates(["roomid","orderdate"],keep="first")
print data.shape
data["orderdate"]=pd.to_datetime(data["orderdate"]).dt.dayofyear

for c in ["roomid"
          ]:
    data[c] = data[c].apply(remove)

data.to_csv("feature_room_30days_realratio_test.csv",index=None)
#==============================================basic_recent3_ordernum_ratio========================================
for i in range(10):
    if i==0:
        data=pd.read_csv("data/train_%s.csv"%i)[["hotelid","basicroomid","orderdate","basic_recent3_ordernum_ratio"]]
    else:
        data=data.append(pd.read_csv("data/train_%s.csv"%i)[["hotelid","basicroomid","orderdate","basic_recent3_ordernum_ratio"]])

data=data.drop_duplicates()
print data.shape
data["orderdate"]=pd.to_datetime(data["orderdate"]).dt.dayofyear

for c in ["hotelid", "basicroomid"
          ]:
    data[c] = data[c].apply(remove)

data.to_csv("feature_3days_train.csv",index=None)

for i in range(10):
    if i==0:
        data=pd.read_csv("data/test_%s.csv"%i)[["hotelid","basicroomid","orderdate","basic_recent3_ordernum_ratio"]]
    else:
        data=data.append(pd.read_csv("data/test_%s.csv"%i)[["hotelid","basicroomid","orderdate","basic_recent3_ordernum_ratio"]])

data=data.drop_duplicates()
print data.shape
data["orderdate"]=pd.to_datetime(data["orderdate"]).dt.dayofyear

for c in ["hotelid", "basicroomid"
          ]:
    data[c] = data[c].apply(remove)

data.to_csv("feature_3days_test.csv",index=None)
#==============================================basic_week_ordernum_ratio========================================
for i in range(10):
    if i==0:
        data=pd.read_csv("data/train_%s.csv"%i)[["hotelid","basicroomid","orderdate","basic_week_ordernum_ratio"]]
    else:
        data=data.append(pd.read_csv("data/train_%s.csv"%i)[["hotelid","basicroomid","orderdate","basic_week_ordernum_ratio"]])

data=data.drop_duplicates()
print data.shape
data["orderdate"]=pd.to_datetime(data["orderdate"]).dt.dayofyear

for c in ["hotelid", "basicroomid"
          ]:
    data[c] = data[c].apply(remove)

data.to_csv("feature_basic_7days_train.csv",index=None)

for i in range(10):
    if i==0:
        data=pd.read_csv("data/test_%s.csv"%i)[["hotelid","basicroomid","orderdate","basic_week_ordernum_ratio"]]
    else:
        data=data.append(pd.read_csv("data/test_%s.csv"%i)[["hotelid","basicroomid","orderdate","basic_week_ordernum_ratio"]])

data=data.drop_duplicates()
print data.shape
data["orderdate"]=pd.to_datetime(data["orderdate"]).dt.dayofyear

for c in ["hotelid", "basicroomid"
          ]:
    data[c] = data[c].apply(remove)

data.to_csv("feature_basic_7days_test.csv",index=None)
#==============================================basic_30days_ordnumratio========================================
for i in range(10):
    if i==0:
        data=pd.read_csv("data/train_%s.csv"%i)[["hotelid","basicroomid","orderdate","basic_30days_ordnumratio"]]
    else:
        data=data.append(pd.read_csv("data/train_%s.csv"%i)[["hotelid","basicroomid","orderdate","basic_30days_ordnumratio"]])

data=data.sort_values("basic_30days_ordnumratio")
data=data.drop_duplicates(["hotelid","basicroomid","orderdate"],keep="first")
print data.shape
data["orderdate"]=pd.to_datetime(data["orderdate"]).dt.dayofyear

for c in ["hotelid", "basicroomid"
          ]:
    data[c] = data[c].apply(remove)

data.to_csv("feature_basic_30days_train.csv",index=None)

for i in range(10):
    if i==0:
        data=pd.read_csv("data/test_%s.csv"%i)[["hotelid","basicroomid","orderdate","basic_30days_ordnumratio"]]
    else:
        data=data.append(pd.read_csv("data/test_%s.csv"%i)[["hotelid","basicroomid","orderdate","basic_30days_ordnumratio"]])

data=data.sort_values("basic_30days_ordnumratio")
data=data.drop_duplicates(["hotelid","basicroomid","orderdate"],keep="first")
print data.shape
data["orderdate"]=pd.to_datetime(data["orderdate"]).dt.dayofyear

for c in ["hotelid", "basicroomid"
          ]:
    data[c] = data[c].apply(remove)

data.to_csv("feature_basic_30days_test.csv",index=None)
#==============================================room_30days_ordnumratio========================================
for i in range(10):
    if i==0:
        data=pd.read_csv("data/train_%s.csv"%i)[["hotelid","roomid","orderdate","room_30days_ordnumratio"]]
    else:
        data=data.append(pd.read_csv("data/train_%s.csv"%i)[["hotelid","roomid","orderdate","room_30days_ordnumratio"]])

data=data.drop_duplicates()
print data.shape
data["orderdate"]=pd.to_datetime(data["orderdate"]).dt.dayofyear

for c in ["hotelid", "roomid"
          ]:
    data[c] = data[c].apply(remove)

data.to_csv("feature_room_30days_ordnumratio_train.csv",index=None)

for i in range(10):
    if i==0:
        data=pd.read_csv("data/test_%s.csv"%i)[["hotelid","roomid","orderdate","room_30days_ordnumratio"]]
    else:
        data=data.append(pd.read_csv("data/test_%s.csv"%i)[["hotelid","roomid","orderdate","room_30days_ordnumratio"]])

data=data.drop_duplicates()
print data.shape
data["orderdate"]=pd.to_datetime(data["orderdate"]).dt.dayofyear

for c in ["hotelid", "roomid"
          ]:
    data[c] = data[c].apply(remove)

data.to_csv("feature_room_30days_ordnumratio_test.csv",index=None)