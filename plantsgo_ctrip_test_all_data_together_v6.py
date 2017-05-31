#encoding=utf8
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,GradientBoostingRegressor
from sklearn import preprocessing
import lightgbm as lgb
import re

def remove(x):
    try:
        return int(re.search("\d+", x).group())
    except:
        return 0

def clean(all):
    dt1=pd.to_datetime(all["orderdate"])
    dt2=pd.to_datetime(all["orderdate_lastord"])
    all["orderdate"]=dt1.dt.weekday
    all["orderdate_lastord"]=dt2.dt.weekday
    for c in ["orderid","uid","hotelid","basicroomid","roomid",
        "orderid_lastord","hotelid_lastord","roomid_lastord","basicroomid_lastord",
              ]:
        all[c] = all[c].apply(remove)
    return all




# 每个basicid价格的中位数
def df_median(df):
    add = pd.DataFrame(df.groupby(["orderid", "basicroomid"]).price_deduct.median()).reset_index()
    add.columns = ["orderid", "basicroomid", "basicroomid_price_deduct_median"]
    df = df.merge(add, on=["orderid", "basicroomid"], how="left")
    return df

# 每个basicid价格的最小值
def df_min(df):
    add = pd.DataFrame(df.groupby(["orderid", "basicroomid"]).price_deduct.min()).reset_index()
    add.columns = ["orderid", "basicroomid", "basicroomid_price_deduct_min"]
    df = df.merge(add, on=["orderid", "basicroomid"], how="left")
    return df

# 每个orderid价格的最小值
def df_min_orderid(df):
    add = pd.DataFrame(df.groupby(["orderid"]).price_deduct.min()).reset_index()
    add.columns = ["orderid", "orderid_price_deduct_min"]
    df = df.merge(add, on=["orderid"], how="left")
    return df

#排序特征
def df_rank_mean(df):
    add = pd.DataFrame(df.groupby(["basicroomid"]).orderid_price_deduct_min_rank.mean()).reset_index()
    add.columns = ["basicroomid","orderid_price_deduct_min_rank_mean"]
    df = df.merge(add, on=["basicroomid"], how="left")
    return df

def df_roomrank_mean(df):
    add = pd.DataFrame(df.groupby(["roomid"]).basicroomid_price_rank.mean()).reset_index()
    add.columns = ["roomid","basicroomid_price_rank_mean"]
    df = df.merge(add, on=["roomid"], how="left")
    return df

#构造特征
all = pd.read_table("ctrip/competition_train.txt")
all = clean(all)

all = df_median(all)
all = df_min(all)
all = df_min_orderid(all)

all["basicroomid_price_rank"] = all['price_deduct'].groupby([all['orderid'], all['basicroomid']]).rank()
all["orderid_price_deduct_min_rank"] = all['orderid_price_deduct_min'].groupby(all['orderid']).rank()

all = df_rank_mean(all)
all = df_roomrank_mean(all)

all["city_num"] = all["user_ordernum"] / all["user_citynum"]
all["area_price"] = all["user_avgprice"] / all["user_avgroomarea"]
all["price_max_min_rt"] = all["user_maxprice"] / all["user_minprice"]
all["basicroomid_price_deduct_min_minprice_rt"] = all["basicroomid_price_deduct_min"] / all["user_minprice"]

all["price_dif"] = all["basicroomid_price_deduct_min"] - all["price_deduct"]
all["price_dif_hotel"] = all["basicroomid_price_deduct_min"] - all["hotel_minprice_lastord"]
all["price_dif_basic"] = all["basicroomid_price_deduct_min"] - all["basic_minprice_lastord"]

all["price_dif_rt"] = all["basicroomid_price_deduct_min"] / all["price_deduct"]
all["price_dif_hotel_rt"] = all["basicroomid_price_deduct_min"] / all["hotel_minprice_lastord"]
all["price_dif_basic_rt"] = all["basicroomid_price_deduct_min"] / all["basic_minprice_lastord"]

all["price_dif_hotel"] = all["orderid_price_deduct_min"] - all["price_deduct"]
all["price_dif_hotel_hotel"] = all["orderid_price_deduct_min"] - all["hotel_minprice_lastord"]
all["price_dif_basic_hotel"] = all["orderid_price_deduct_min"] - all["basic_minprice_lastord"]

all["price_dif_hotel_rt"] = all["orderid_price_deduct_min"] / all["price_deduct"]
all["price_dif_hotel_hotel_rt"] = all["orderid_price_deduct_min"] / all["hotel_minprice_lastord"]
all["price_dif_basic_hotel_rt"] = all["orderid_price_deduct_min"] / all["basic_minprice_lastord"]

# all["order_basic_minprice_dif"]=all["basicroomid_price_deduct_min"]-all["orderid_price_deduct_min"]
all["order_basic_minprice_rt"] = all["basicroomid_price_deduct_min"] / all["orderid_price_deduct_min"]
# all["hotel_basic_minprice_lastord_rt"]=all["basic_minprice_lastord"]/all["hotel_minprice_lastord"]

# 上次订购的价格和当时最低价的比
all["hotel_last_price_min_rt"] = all["price_last_lastord"] / all["hotel_minprice_lastord"]
all["basic_last_price_min_rt"] = all["price_last_lastord"] / all["basic_minprice_lastord"]
all["hotel_last_price_min_dif"] = all["price_last_lastord"] - all["hotel_minprice_lastord"]
all["basic_last_price_min_dif"] = all["price_last_lastord"] - all["basic_minprice_lastord"]

all["price_tail1"] = all["price_deduct"] % 10
all["price_tail1"] = map(lambda x: 1 if x == 4 or x == 7 else 0, all["price_tail1"])
# all["price_tail2"]=all["price_deduct"]%100
all["basic_equal"] = map(lambda x, y: 1 if x == y else 0, all["basicroomid"], all["basicroomid_lastord"])
# del all["basicroomid_lastord"]
all["room_equal"] = map(lambda x, y: 1 if x == y else 0, all["roomid"], all["roomid_lastord"])
# del all["roomid_lastord"]
all["hotel_equal"] = map(lambda x, y: 1 if x == y else 0, all["hotelid"], all["hotelid_lastord"])
# del all["hotelid_lastord"]
all["rank_equal"] = map(lambda x, y: 1 if x == y else 0, all["rank"], all["rank_lastord"])

# 价格高低
all["price_dx"] = map(lambda x, y: x - y, all["price_deduct"], all["price_last_lastord"])

all["return_dx"] = map(lambda x, y: x - y, all["returnvalue"], all["return_lastord"])

all["price_ori"] = map(lambda x, y: x + y, all["price_deduct"], all["returnvalue"])

for i in [2, 3, 4, 5, 6, 8]:
    all["service_equal_%s" % i] = map(lambda x, y: 1 if x == y else 0, all["roomservice_%s" % i],
                                      all["roomservice_%s_lastord" % i])
    del all["roomservice_%s_lastord" % i]

for i in [2, 3, 4, 5, 6]:
    all["roomtag_equal_%s" % i] = map(lambda x, y: 1 if x == y else 0, all["roomtag_%s" % i],
                                      all["roomtag_%s_lastord" % i])
    del all["roomtag_%s_lastord" % i]

for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    all["ordertype_%s_num" % i] = map(lambda x, y: x * y, all["ordertype_%s_ratio" % i], all["user_ordernum"])
    del all["ordertype_%s_ratio" % i]

# 所有的
for c in ["orderbehavior_1_ratio", "orderbehavior_2_ratio", "orderbehavior_6_ratio", "orderbehavior_7_ratio",
          # "user_roomservice_4_0ratio","user_roomservice_4_1ratio","user_roomservice_4_2ratio","user_roomservice_4_3ratio","user_roomservice_4_4ratio","user_roomservice_4_5ratio","user_roomservice_3_123ratio","user_roomservice_6_2ratio","user_roomservice_6_1ratio","user_roomservice_6_0ratio","user_roomservice_5_1ratio","user_roomservice_7_0ratio","user_roomservice_2_1ratio","user_roomservice_8_1ratio","user_roomservice_5_345ratio"

          ]:
    all[c] = map(lambda x, y: x * y, all[c], all["user_ordernum"])

# 一周的
for c in ["orderbehavior_3_ratio_1week", "orderbehavior_4_ratio_1week", "orderbehavior_5_ratio_1week",
          # "user_roomservice_3_123ratio_1week","user_roomservice_7_1ratio_1week","user_roomservice_7_0ratio_1week","user_roomservice_4_5ratio_1week","user_roomservice_4_4ratio_1week","user_roomservice_4_2ratio_1week","user_roomservice_4_3ratio_1week","user_roomservice_4_0ratio_1week"
          ]:
    all[c] = map(lambda x, y: x * y, all[c], all["user_ordnum_1week"])

# 一个月的
for c in ["orderbehavior_3_ratio_1month", "orderbehavior_4_ratio_1month", "orderbehavior_5_ratio_1month",
          # "user_roomservice_3_123ratio_1month", "user_roomservice_7_1ratio_1month", "user_roomservice_7_0ratio_1month","user_roomservice_4_5ratio_1month", "user_roomservice_4_4ratio_1month", "user_roomservice_4_2ratio_1month","user_roomservice_4_3ratio_1month", "user_roomservice_4_0ratio_1month"

          ]:
    all[c] = map(lambda x, y: x * y, all[c], all["user_ordnum_1month"])

# 三个月的
for c in ["orderbehavior_3_ratio_3month", "orderbehavior_4_ratio_3month", "orderbehavior_5_ratio_3month",
          # "user_roomservice_3_123ratio_3month", "user_roomservice_7_1ratio_3month", "user_roomservice_7_0ratio_3month","user_roomservice_4_5ratio_3month", "user_roomservice_4_4ratio_3month", "user_roomservice_4_2ratio_3month","user_roomservice_4_3ratio_3month", "user_roomservice_4_0ratio_3month"

          ]:
    all[c] = map(lambda x, y: x * y, all[c], all["user_ordnum_3month"])

all["price_star"] = all["price_deduct"] / (all["star"] - 1)
all["price_minarea"] = all["price_deduct"] / (all["basic_minarea"] - 1)

all["star_dif"] = all["user_avgstar"] - all["star"]

all["price_ave_dif_rt"] = all["price_deduct"] / all["user_avgdealprice"]
all["price_ave_star_dif"] = all["price_deduct"] / all["user_avgprice_star"]
all["price_h_w_rt"] = all["user_avgdealpriceholiday"] / all["user_avgdealpriceworkday"]

all["price_ave_dif"] = all["price_deduct"] - all["user_avgdealprice"]

all["user_roomservice_4_32_rt"] = all["user_roomservice_4_3ratio"] / all["user_roomservice_4_2ratio"]
all["user_roomservice_4_43_rt"] = all["user_roomservice_4_4ratio"] / all["user_roomservice_4_3ratio"]

print all.head()

train=all

#算法测试
train_x=train.drop("orderlabel",axis=1)#.values
train_y=train["orderlabel"]

print train_x.shape
#lgb算法
train_matrix = lgb.Dataset(train_x, label=train_y)
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'min_child_weight': 1.5,
    'num_leaves': 2 ** 5,
    'lambda_l2': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'learning_rate': 0.05,
    'tree_method': 'exact',
    'seed': 2017,
    'nthread': 12,
    'silent': True,
}
num_round = 1300
model = lgb.train(params, train_matrix, num_round,
                      )

#构造特征
all = pd.read_table("ctrip/competition_test.txt")
all = clean(all)

all = df_median(all)
all = df_min(all)
all = df_min_orderid(all)

all["basicroomid_price_rank"] = all['price_deduct'].groupby([all['orderid'], all['basicroomid']]).rank()
all["orderid_price_deduct_min_rank"] = all['orderid_price_deduct_min'].groupby(all['orderid']).rank()

all = df_rank_mean(all)
all = df_roomrank_mean(all)

all["city_num"] = all["user_ordernum"] / all["user_citynum"]
all["area_price"] = all["user_avgprice"] / all["user_avgroomarea"]
all["price_max_min_rt"] = all["user_maxprice"] / all["user_minprice"]
all["basicroomid_price_deduct_min_minprice_rt"] = all["basicroomid_price_deduct_min"] / all["user_minprice"]

all["price_dif"] = all["basicroomid_price_deduct_min"] - all["price_deduct"]
all["price_dif_hotel"] = all["basicroomid_price_deduct_min"] - all["hotel_minprice_lastord"]
all["price_dif_basic"] = all["basicroomid_price_deduct_min"] - all["basic_minprice_lastord"]

all["price_dif_rt"] = all["basicroomid_price_deduct_min"] / all["price_deduct"]
all["price_dif_hotel_rt"] = all["basicroomid_price_deduct_min"] / all["hotel_minprice_lastord"]
all["price_dif_basic_rt"] = all["basicroomid_price_deduct_min"] / all["basic_minprice_lastord"]

all["price_dif_hotel"] = all["orderid_price_deduct_min"] - all["price_deduct"]
all["price_dif_hotel_hotel"] = all["orderid_price_deduct_min"] - all["hotel_minprice_lastord"]
all["price_dif_basic_hotel"] = all["orderid_price_deduct_min"] - all["basic_minprice_lastord"]

all["price_dif_hotel_rt"] = all["orderid_price_deduct_min"] / all["price_deduct"]
all["price_dif_hotel_hotel_rt"] = all["orderid_price_deduct_min"] / all["hotel_minprice_lastord"]
all["price_dif_basic_hotel_rt"] = all["orderid_price_deduct_min"] / all["basic_minprice_lastord"]

# all["order_basic_minprice_dif"]=all["basicroomid_price_deduct_min"]-all["orderid_price_deduct_min"]
all["order_basic_minprice_rt"] = all["basicroomid_price_deduct_min"] / all["orderid_price_deduct_min"]
# all["hotel_basic_minprice_lastord_rt"]=all["basic_minprice_lastord"]/all["hotel_minprice_lastord"]

# 上次订购的价格和当时最低价的比
all["hotel_last_price_min_rt"] = all["price_last_lastord"] / all["hotel_minprice_lastord"]
all["basic_last_price_min_rt"] = all["price_last_lastord"] / all["basic_minprice_lastord"]
all["hotel_last_price_min_dif"] = all["price_last_lastord"] - all["hotel_minprice_lastord"]
all["basic_last_price_min_dif"] = all["price_last_lastord"] - all["basic_minprice_lastord"]

all["price_tail1"] = all["price_deduct"] % 10
all["price_tail1"] = map(lambda x: 1 if x == 4 or x == 7 else 0, all["price_tail1"])
# all["price_tail2"]=all["price_deduct"]%100
all["basic_equal"] = map(lambda x, y: 1 if x == y else 0, all["basicroomid"], all["basicroomid_lastord"])
# del all["basicroomid_lastord"]
all["room_equal"] = map(lambda x, y: 1 if x == y else 0, all["roomid"], all["roomid_lastord"])
# del all["roomid_lastord"]
all["hotel_equal"] = map(lambda x, y: 1 if x == y else 0, all["hotelid"], all["hotelid_lastord"])
# del all["hotelid_lastord"]
all["rank_equal"] = map(lambda x, y: 1 if x == y else 0, all["rank"], all["rank_lastord"])

# 价格高低
all["price_dx"] = map(lambda x, y: x - y, all["price_deduct"], all["price_last_lastord"])

all["return_dx"] = map(lambda x, y: x - y, all["returnvalue"], all["return_lastord"])

all["price_ori"] = map(lambda x, y: x + y, all["price_deduct"], all["returnvalue"])

for i in [2, 3, 4, 5, 6, 8]:
    all["service_equal_%s" % i] = map(lambda x, y: 1 if x == y else 0, all["roomservice_%s" % i],
                                      all["roomservice_%s_lastord" % i])
    del all["roomservice_%s_lastord" % i]

for i in [2, 3, 4, 5, 6]:
    all["roomtag_equal_%s" % i] = map(lambda x, y: 1 if x == y else 0, all["roomtag_%s" % i],
                                      all["roomtag_%s_lastord" % i])
    del all["roomtag_%s_lastord" % i]

for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    all["ordertype_%s_num" % i] = map(lambda x, y: x * y, all["ordertype_%s_ratio" % i], all["user_ordernum"])
    del all["ordertype_%s_ratio" % i]

# 所有的
for c in ["orderbehavior_1_ratio", "orderbehavior_2_ratio", "orderbehavior_6_ratio", "orderbehavior_7_ratio",
          # "user_roomservice_4_0ratio","user_roomservice_4_1ratio","user_roomservice_4_2ratio","user_roomservice_4_3ratio","user_roomservice_4_4ratio","user_roomservice_4_5ratio","user_roomservice_3_123ratio","user_roomservice_6_2ratio","user_roomservice_6_1ratio","user_roomservice_6_0ratio","user_roomservice_5_1ratio","user_roomservice_7_0ratio","user_roomservice_2_1ratio","user_roomservice_8_1ratio","user_roomservice_5_345ratio"

          ]:
    all[c] = map(lambda x, y: x * y, all[c], all["user_ordernum"])

# 一周的
for c in ["orderbehavior_3_ratio_1week", "orderbehavior_4_ratio_1week", "orderbehavior_5_ratio_1week",
          # "user_roomservice_3_123ratio_1week","user_roomservice_7_1ratio_1week","user_roomservice_7_0ratio_1week","user_roomservice_4_5ratio_1week","user_roomservice_4_4ratio_1week","user_roomservice_4_2ratio_1week","user_roomservice_4_3ratio_1week","user_roomservice_4_0ratio_1week"
          ]:
    all[c] = map(lambda x, y: x * y, all[c], all["user_ordnum_1week"])

# 一个月的
for c in ["orderbehavior_3_ratio_1month", "orderbehavior_4_ratio_1month", "orderbehavior_5_ratio_1month",
          # "user_roomservice_3_123ratio_1month", "user_roomservice_7_1ratio_1month", "user_roomservice_7_0ratio_1month","user_roomservice_4_5ratio_1month", "user_roomservice_4_4ratio_1month", "user_roomservice_4_2ratio_1month","user_roomservice_4_3ratio_1month", "user_roomservice_4_0ratio_1month"

          ]:
    all[c] = map(lambda x, y: x * y, all[c], all["user_ordnum_1month"])

# 三个月的
for c in ["orderbehavior_3_ratio_3month", "orderbehavior_4_ratio_3month", "orderbehavior_5_ratio_3month",
          # "user_roomservice_3_123ratio_3month", "user_roomservice_7_1ratio_3month", "user_roomservice_7_0ratio_3month","user_roomservice_4_5ratio_3month", "user_roomservice_4_4ratio_3month", "user_roomservice_4_2ratio_3month","user_roomservice_4_3ratio_3month", "user_roomservice_4_0ratio_3month"

          ]:
    all[c] = map(lambda x, y: x * y, all[c], all["user_ordnum_3month"])

all["price_star"] = all["price_deduct"] / (all["star"] - 1)
all["price_minarea"] = all["price_deduct"] / (all["basic_minarea"] - 1)

all["star_dif"] = all["user_avgstar"] - all["star"]

all["price_ave_dif_rt"] = all["price_deduct"] / all["user_avgdealprice"]
all["price_ave_star_dif"] = all["price_deduct"] / all["user_avgprice_star"]
all["price_h_w_rt"] = all["user_avgdealpriceholiday"] / all["user_avgdealpriceworkday"]

all["price_ave_dif"] = all["price_deduct"] - all["user_avgdealprice"]

all["user_roomservice_4_32_rt"] = all["user_roomservice_4_3ratio"] / all["user_roomservice_4_2ratio"]
all["user_roomservice_4_43_rt"] = all["user_roomservice_4_4ratio"] / all["user_roomservice_4_3ratio"]

print all.shape
print all.head()

online = model.predict(all.values)
online = pd.DataFrame(online)
online.columns = ["prob"]
online["orderid"] = all["orderid"].values
online["predict_roomid"] = all["roomid"].values

result=online

result.to_csv("all_result_together_v6.csv",index=None)
result = result.sort_values("prob")
del result["prob"]
result = result.drop_duplicates("orderid", keep="last")
result["orderid"]=result["orderid"].apply(lambda x:"ORDER_"+str(x))
result["predict_roomid"]=result["predict_roomid"].apply(lambda x:"ROOM_"+str(x))
result.to_csv("sub_all_result_together_v6.csv",index=None)





