#encoding=utf8
import pandas as pd
import lightgbm as lgb
from com_uitl import *


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


#添加转化率特征
leak=pd.read_csv("try/leak.csv")
#提取basicroomid的转化率
feature_df=leak[["orderid","basicroomid","orderlabel"]].copy()
feature_df.sort_values("orderlabel")
feature_df=feature_df.drop_duplicates(["orderid","basicroomid"],keep="last")
basicroom_mean=pd.DataFrame(feature_df.groupby("basicroomid").orderlabel.mean()).reset_index()
basicroom_mean.columns=["basicroomid","basicroomid_mean"]

basicroom_sum=pd.DataFrame(feature_df.groupby("basicroomid").orderlabel.sum()).reset_index()
basicroom_sum.columns=["basicroomid","basicroomid_sum"]

del leak

#basic的trick
trick_1=pd.read_csv("feature_3days_train.csv").append(pd.read_csv("feature_3days_test.csv"))
trick_2=trick_1.copy()
trick_3=trick_1.copy()
trick_1.columns=["hotelid","basicroomid","orderdate","trick_1"]
trick_2.columns=["hotelid","basicroomid","orderdate","trick_2"]
trick_3.columns=["hotelid","basicroomid","orderdate","trick_3"]
trick_1["orderdate"]=trick_1["orderdate"]-1
trick_2["orderdate"]=trick_2["orderdate"]-2
trick_3["orderdate"]=trick_3["orderdate"]-3
#room的trick
trick_room=pd.read_csv("feature_room_30days_ordnumratio_train.csv").append(pd.read_csv("feature_room_30days_ordnumratio_test.csv"))
trick_room.columns=["hotelid","roomid","orderdate","trick_room"]
trick_room["orderdate"]=trick_room["orderdate"]-1
#basic七天的的trick
trick_basic_7d=pd.read_csv("feature_basic_7days_train.csv").append(pd.read_csv("feature_basic_7days_test.csv"))
trick_basic_7d.columns=["hotelid","basicroomid","orderdate","trick_basic_7d"]
trick_basic_7d["orderdate"]=trick_basic_7d["orderdate"]-1
#basic30天的trick
trick_basic_30d=pd.read_csv("feature_basic_30days_train.csv").append(pd.read_csv("feature_basic_30days_test.csv"))
trick_basic_30d_fill=trick_basic_30d.copy()
trick_basic_30d.columns=["hotelid","basicroomid","orderdate","trick_basic_30d"]
trick_basic_30d["orderdate"]=trick_basic_30d["orderdate"]-1
#trick_basic_realratio
trick_basic_realratio=pd.read_csv("feature_basic_30days_realratio_train.csv").append(pd.read_csv("feature_basic_30days_realratio_test.csv"))
trick_basic_realratio.columns=["basicroomid","orderdate","trick_basic_30days_realratio"]
trick_basic_realratio["orderdate"]=trick_basic_realratio["orderdate"]-1
#trick_room_realratio
trick_room_realratio=pd.read_csv("feature_room_30days_realratio_train.csv").append(pd.read_csv("feature_room_30days_realratio_test.csv"))
trick_room_realratio.columns=["roomid","orderdate","trick_room_30days_realratio"]
trick_room_realratio["orderdate"]=trick_room_realratio["orderdate"]-1


#构造特征
#for j in range(10):
for j in [0,2,4,6,8]:
#for j in [1,3,5,7,9]:
    print j
    all=pd.read_csv("try/offline_%s.csv"%j)

    # test中的basic_30days_ordnumratio数据异常，需要重新设置
    del all["basic_30days_ordnumratio"]
    all = all.merge(trick_basic_30d_fill, on=["hotelid", "basicroomid", "orderdate"], how="left")
    # 使用穿越特征
    all = all.merge(trick_1, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all = all.merge(trick_2, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all = all.merge(trick_3, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all["basic_trick_today"] = all["trick_1"] - all["basic_recent3_ordernum_ratio"]

    all = all.merge(trick_room, on=["hotelid", "roomid", "orderdate"], how="left")
    all["room_trick_today"] = all["trick_room"] - all["room_30days_ordnumratio"]

    all = all.merge(trick_basic_7d, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all["basic_7d_trick_today"] = all["trick_basic_7d"] - all["basic_week_ordernum_ratio"]

    all = all.merge(trick_basic_30d, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all["basic_30d_trick_today"] = all["trick_basic_30d"] - all["basic_30days_ordnumratio"]

    all = all.merge(trick_basic_realratio, on=["basicroomid", "orderdate"], how="left")
    all["basic_30days_realratio_today"] = all["trick_basic_30days_realratio"] - all["basic_30days_realratio"]
    all = all.merge(trick_room_realratio, on=["roomid", "orderdate"], how="left")
    all["room_30days_realratio_today"] = all["trick_room_30days_realratio"] - all["room_30days_realratio"]

    all["roomid_ori"] = all["roomid"]
    all["roomid"] = map(lambda x, y: int(str(x)[:-len(str(y))]), all["roomid"], all["rank"])

    all = all.merge(basicroom_mean, on="basicroomid", how="left").fillna(0)
    all = all.merge(basicroom_sum, on="basicroomid", how="left").fillna(0)

    all=df_median(all)
    all=df_min(all)
    all=df_min_orderid(all)

    all["basicroomid_price_rank"] = all['price_deduct'].groupby([all['orderid'], all['basicroomid']]).rank()
    all["orderid_price_deduct_min_rank"] = all['orderid_price_deduct_min'].groupby(all['orderid']).rank()

    all = df_rank_mean(all)
    all = df_roomrank_mean(all)


    #添加新特征20170527
    #平均值
    all=merge_mean(all,["basicroomid"],"basic_week_ordernum_ratio","basic_week_ordernum_ratio_mean")
    all=merge_mean(all,["basicroomid"],"basic_recent3_ordernum_ratio","basic_recent3_ordernum_ratio_mean")
    all=merge_mean(all,["basicroomid"],"basic_comment_ratio","basic_comment_ratio_mean")
    all=merge_mean(all,["basicroomid"],"basic_30days_ordnumratio","basic_30days_ordnumratio_mean")
    all=merge_mean(all,["basicroomid"],"basic_30days_realratio","basic_30days_realratio_mean")
    all=merge_mean(all,["roomid"],"room_30days_ordnumratio","room_30days_ordnumratio_mean")
    all=merge_mean(all,["roomid"],"room_30days_realratio","room_30days_realratio_mean")


    all["city_num"]=all["user_ordernum"]/all["user_citynum"]
    all["area_price"]=all["user_avgprice"]/all["user_avgroomarea"]
    all["price_max_min_rt"]=all["user_maxprice"]/all["user_minprice"]
    all["basicroomid_price_deduct_min_minprice_rt"]=all["basicroomid_price_deduct_min"]/all["user_minprice"]

    all["price_dif"]=all["basicroomid_price_deduct_min"]-all["price_deduct"]
    all["price_dif_hotel"]=all["basicroomid_price_deduct_min"]-all["hotel_minprice_lastord"]
    all["price_dif_basic"]=all["basicroomid_price_deduct_min"]-all["basic_minprice_lastord"]

    all["price_dif_rt"]=all["basicroomid_price_deduct_min"]/all["price_deduct"]
    all["price_dif_hotel_rt"]=all["basicroomid_price_deduct_min"]/all["hotel_minprice_lastord"]
    all["price_dif_basic_rt"]=all["basicroomid_price_deduct_min"]/all["basic_minprice_lastord"]

    all["price_dif_hotel"]=all["orderid_price_deduct_min"]-all["price_deduct"]
    all["price_dif_hotel_hotel"]=all["orderid_price_deduct_min"]-all["hotel_minprice_lastord"]
    all["price_dif_basic_hotel"]=all["orderid_price_deduct_min"]-all["basic_minprice_lastord"]

    all["price_dif_hotel_rt"]=all["orderid_price_deduct_min"]/all["price_deduct"]
    all["price_dif_hotel_hotel_rt"]=all["orderid_price_deduct_min"]/all["hotel_minprice_lastord"]
    all["price_dif_basic_hotel_rt"]=all["orderid_price_deduct_min"]/all["basic_minprice_lastord"]

    #all["order_basic_minprice_dif"]=all["basicroomid_price_deduct_min"]-all["orderid_price_deduct_min"]
    all["order_basic_minprice_rt"]=all["basicroomid_price_deduct_min"]/all["orderid_price_deduct_min"]
    #all["hotel_basic_minprice_lastord_rt"]=all["basic_minprice_lastord"]/all["hotel_minprice_lastord"]

    #上次订购的价格和当时最低价的比
    all["hotel_last_price_min_rt"]=all["price_last_lastord"]/all["hotel_minprice_lastord"]
    all["basic_last_price_min_rt"]=all["price_last_lastord"]/all["basic_minprice_lastord"]
    all["hotel_last_price_min_dif"]=all["price_last_lastord"]-all["hotel_minprice_lastord"]
    all["basic_last_price_min_dif"]=all["price_last_lastord"]-all["basic_minprice_lastord"]


    all["price_tail1"]=all["price_deduct"]%10
    all["price_tail1"]=map(lambda x:1 if x==4 or x==7 else 0,all["price_tail1"])
    #all["price_tail2"]=all["price_deduct"]%100
    all["basic_equal"]=map(lambda x,y:1 if x==y else 0,all["basicroomid"],all["basicroomid_lastord"])
    #del all["basicroomid_lastord"]
    all["room_equal"]=map(lambda x,y:1 if x==y else 0,all["roomid"],all["roomid_lastord"])
    #del all["roomid_lastord"]
    all["hotel_equal"]=map(lambda x,y:1 if x==y else 0,all["hotelid"],all["hotelid_lastord"])
    #del all["hotelid_lastord"]
    all["rank_equal"]=map(lambda x,y:1 if x==y else 0,all["rank"],all["rank_lastord"])

    #价格高低
    all["price_dx"] = map(lambda x, y: x-y, all["price_deduct"], all["price_last_lastord"])

    all["return_dx"] = map(lambda x, y: x-y, all["returnvalue"], all["return_lastord"])

    all["price_ori"] = map(lambda x, y:x+y, all["price_deduct"], all["returnvalue"])


    for i in [2,3,4,5,6,8]:
        all["service_equal_%s"%i] = map(lambda x, y: 1 if x == y else 0, all["roomservice_%s"%i], all["roomservice_%s_lastord"%i])
        del all["roomservice_%s_lastord"%i]

    for i in [2,3,4,5,6]:
        all["roomtag_equal_%s"%i] = map(lambda x, y: 1 if x == y else 0, all["roomtag_%s"%i], all["roomtag_%s_lastord"%i])
        del all["roomtag_%s_lastord"%i]

    for i in [1,2,3,4,5,6,7,8,9,10,11]:
        all["ordertype_%s_num"%i] = map(lambda x, y:x*y, all["ordertype_%s_ratio"%i], all["user_ordernum"])
        del all["ordertype_%s_ratio"%i]

    #所有的
    for c in ["orderbehavior_1_ratio","orderbehavior_2_ratio","orderbehavior_6_ratio","orderbehavior_7_ratio",
               #"user_roomservice_4_0ratio","user_roomservice_4_1ratio","user_roomservice_4_2ratio","user_roomservice_4_3ratio","user_roomservice_4_4ratio","user_roomservice_4_5ratio","user_roomservice_3_123ratio","user_roomservice_6_2ratio","user_roomservice_6_1ratio","user_roomservice_6_0ratio","user_roomservice_5_1ratio","user_roomservice_7_0ratio","user_roomservice_2_1ratio","user_roomservice_8_1ratio","user_roomservice_5_345ratio"

              ]:
        all[c]=map(lambda x,y:x*y,all[c],all["user_ordernum"])

    #一周的
    for c in ["orderbehavior_3_ratio_1week","orderbehavior_4_ratio_1week","orderbehavior_5_ratio_1week",
              #"user_roomservice_3_123ratio_1week","user_roomservice_7_1ratio_1week","user_roomservice_7_0ratio_1week","user_roomservice_4_5ratio_1week","user_roomservice_4_4ratio_1week","user_roomservice_4_2ratio_1week","user_roomservice_4_3ratio_1week","user_roomservice_4_0ratio_1week"
              ]:
        all[c] = map(lambda x,y: x * y, all[c], all["user_ordnum_1week"])

    #一个月的
    for c in ["orderbehavior_3_ratio_1month","orderbehavior_4_ratio_1month","orderbehavior_5_ratio_1month",
              #"user_roomservice_3_123ratio_1month", "user_roomservice_7_1ratio_1month", "user_roomservice_7_0ratio_1month","user_roomservice_4_5ratio_1month", "user_roomservice_4_4ratio_1month", "user_roomservice_4_2ratio_1month","user_roomservice_4_3ratio_1month", "user_roomservice_4_0ratio_1month"

              ]:
        all[c] = map(lambda x,y: x * y, all[c], all["user_ordnum_1month"])

    #三个月的
    for c in ["orderbehavior_3_ratio_3month","orderbehavior_4_ratio_3month","orderbehavior_5_ratio_3month",
              #"user_roomservice_3_123ratio_3month", "user_roomservice_7_1ratio_3month", "user_roomservice_7_0ratio_3month","user_roomservice_4_5ratio_3month", "user_roomservice_4_4ratio_3month", "user_roomservice_4_2ratio_3month","user_roomservice_4_3ratio_3month", "user_roomservice_4_0ratio_3month"

              ]:
        all[c] = map(lambda x,y: x * y, all[c], all["user_ordnum_3month"])


    all["price_star"]=all["price_deduct"]/(all["star"]-1)
    all["price_minarea"]=all["price_deduct"]/(all["basic_minarea"]-1)

    all["star_dif"]=all["user_avgstar"]-all["star"]

    all["price_ave_dif_rt"]=all["price_deduct"]/all["user_avgdealprice"]
    all["price_ave_star_dif"]=all["price_deduct"]/all["user_avgprice_star"]
    all["price_h_w_rt"]=all["user_avgdealpriceholiday"]/all["user_avgdealpriceworkday"]

    all["price_ave_dif"] = all["price_deduct"] - all["user_avgdealprice"]

    all["user_roomservice_4_32_rt"]=all["user_roomservice_4_3ratio"]/all["user_roomservice_4_2ratio"]
    all["user_roomservice_4_43_rt"]=all["user_roomservice_4_4ratio"]/all["user_roomservice_4_3ratio"]

    print all.shape

    if j==0 or j==1:
        train=all
    else:
        train=train.append(all)

#算法测试
train_y=train["orderlabel"].values
del train["orderlabel"]


print train.shape
#lgb算法
train = lgb.Dataset(train, label=train_y)
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
model = lgb.train(params, train, num_round,
                      )


for j in range(10):
    print j
    all=pd.read_csv("try/online_%s.csv"%j)

    # test中的basic_30days_ordnumratio数据异常，需要重新设置
    del all["basic_30days_ordnumratio"]
    all = all.merge(trick_basic_30d_fill, on=["hotelid", "basicroomid", "orderdate"], how="left")
    # 使用穿越特征
    all = all.merge(trick_1, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all = all.merge(trick_2, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all = all.merge(trick_3, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all["basic_trick_today"] = all["trick_1"] - all["basic_recent3_ordernum_ratio"]

    all = all.merge(trick_room, on=["hotelid", "roomid", "orderdate"], how="left")
    all["room_trick_today"] = all["trick_room"] - all["room_30days_ordnumratio"]

    all = all.merge(trick_basic_7d, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all["basic_7d_trick_today"] = all["trick_basic_7d"] - all["basic_week_ordernum_ratio"]

    all = all.merge(trick_basic_30d, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all["basic_30d_trick_today"] = all["trick_basic_30d"] - all["basic_30days_ordnumratio"]

    all = all.merge(trick_basic_realratio, on=["basicroomid", "orderdate"], how="left")
    all["basic_30days_realratio_today"] = all["trick_basic_30days_realratio"] - all["basic_30days_realratio"]
    all = all.merge(trick_room_realratio, on=["roomid", "orderdate"], how="left")
    all["room_30days_realratio_today"] = all["trick_room_30days_realratio"] - all["room_30days_realratio"]

    all["roomid_ori"] = all["roomid"]
    all["roomid"] = map(lambda x, y: int(str(x)[:-len(str(y))]), all["roomid"], all["rank"])

    all = all.merge(basicroom_mean, on="basicroomid", how="left").fillna(0)
    all = all.merge(basicroom_sum, on="basicroomid", how="left").fillna(0)

    all=df_median(all)
    all=df_min(all)
    all=df_min_orderid(all)

    all["basicroomid_price_rank"] = all['price_deduct'].groupby([all['orderid'], all['basicroomid']]).rank()
    all["orderid_price_deduct_min_rank"] = all['orderid_price_deduct_min'].groupby(all['orderid']).rank()

    all = df_rank_mean(all)
    all = df_roomrank_mean(all)

    #添加新特征20170527
    #平均值
    all=merge_mean(all,["basicroomid"],"basic_week_ordernum_ratio","basic_week_ordernum_ratio_mean")
    all=merge_mean(all,["basicroomid"],"basic_recent3_ordernum_ratio","basic_recent3_ordernum_ratio_mean")
    all=merge_mean(all,["basicroomid"],"basic_comment_ratio","basic_comment_ratio_mean")
    all=merge_mean(all,["basicroomid"],"basic_30days_ordnumratio","basic_30days_ordnumratio_mean")
    all=merge_mean(all,["basicroomid"],"basic_30days_realratio","basic_30days_realratio_mean")
    all=merge_mean(all,["roomid"],"room_30days_ordnumratio","room_30days_ordnumratio_mean")
    all=merge_mean(all,["roomid"],"room_30days_realratio","room_30days_realratio_mean")

    all["city_num"]=all["user_ordernum"]/all["user_citynum"]
    all["area_price"]=all["user_avgprice"]/all["user_avgroomarea"]
    all["price_max_min_rt"]=all["user_maxprice"]/all["user_minprice"]
    all["basicroomid_price_deduct_min_minprice_rt"]=all["basicroomid_price_deduct_min"]/all["user_minprice"]

    all["price_dif"]=all["basicroomid_price_deduct_min"]-all["price_deduct"]
    all["price_dif_hotel"]=all["basicroomid_price_deduct_min"]-all["hotel_minprice_lastord"]
    all["price_dif_basic"]=all["basicroomid_price_deduct_min"]-all["basic_minprice_lastord"]

    all["price_dif_rt"]=all["basicroomid_price_deduct_min"]/all["price_deduct"]
    all["price_dif_hotel_rt"]=all["basicroomid_price_deduct_min"]/all["hotel_minprice_lastord"]
    all["price_dif_basic_rt"]=all["basicroomid_price_deduct_min"]/all["basic_minprice_lastord"]

    all["price_dif_hotel"]=all["orderid_price_deduct_min"]-all["price_deduct"]
    all["price_dif_hotel_hotel"]=all["orderid_price_deduct_min"]-all["hotel_minprice_lastord"]
    all["price_dif_basic_hotel"]=all["orderid_price_deduct_min"]-all["basic_minprice_lastord"]

    all["price_dif_hotel_rt"]=all["orderid_price_deduct_min"]/all["price_deduct"]
    all["price_dif_hotel_hotel_rt"]=all["orderid_price_deduct_min"]/all["hotel_minprice_lastord"]
    all["price_dif_basic_hotel_rt"]=all["orderid_price_deduct_min"]/all["basic_minprice_lastord"]

    #all["order_basic_minprice_dif"]=all["basicroomid_price_deduct_min"]-all["orderid_price_deduct_min"]
    all["order_basic_minprice_rt"]=all["basicroomid_price_deduct_min"]/all["orderid_price_deduct_min"]
    #all["hotel_basic_minprice_lastord_rt"]=all["basic_minprice_lastord"]/all["hotel_minprice_lastord"]

    #上次订购的价格和当时最低价的比
    all["hotel_last_price_min_rt"]=all["price_last_lastord"]/all["hotel_minprice_lastord"]
    all["basic_last_price_min_rt"]=all["price_last_lastord"]/all["basic_minprice_lastord"]
    all["hotel_last_price_min_dif"]=all["price_last_lastord"]-all["hotel_minprice_lastord"]
    all["basic_last_price_min_dif"]=all["price_last_lastord"]-all["basic_minprice_lastord"]


    all["price_tail1"]=all["price_deduct"]%10
    all["price_tail1"]=map(lambda x:1 if x==4 or x==7 else 0,all["price_tail1"])
    #all["price_tail2"]=all["price_deduct"]%100
    all["basic_equal"]=map(lambda x,y:1 if x==y else 0,all["basicroomid"],all["basicroomid_lastord"])
    #del all["basicroomid_lastord"]
    all["room_equal"]=map(lambda x,y:1 if x==y else 0,all["roomid"],all["roomid_lastord"])
    #del all["roomid_lastord"]
    all["hotel_equal"]=map(lambda x,y:1 if x==y else 0,all["hotelid"],all["hotelid_lastord"])
    #del all["hotelid_lastord"]
    all["rank_equal"]=map(lambda x,y:1 if x==y else 0,all["rank"],all["rank_lastord"])

    #价格高低
    all["price_dx"] = map(lambda x, y: x-y, all["price_deduct"], all["price_last_lastord"])

    all["return_dx"] = map(lambda x, y: x-y, all["returnvalue"], all["return_lastord"])

    all["price_ori"] = map(lambda x, y:x+y, all["price_deduct"], all["returnvalue"])


    for i in [2,3,4,5,6,8]:
        all["service_equal_%s"%i] = map(lambda x, y: 1 if x == y else 0, all["roomservice_%s"%i], all["roomservice_%s_lastord"%i])
        del all["roomservice_%s_lastord"%i]

    for i in [2,3,4,5,6]:
        all["roomtag_equal_%s"%i] = map(lambda x, y: 1 if x == y else 0, all["roomtag_%s"%i], all["roomtag_%s_lastord"%i])
        del all["roomtag_%s_lastord"%i]

    for i in [1,2,3,4,5,6,7,8,9,10,11]:
        all["ordertype_%s_num"%i] = map(lambda x, y:x*y, all["ordertype_%s_ratio"%i], all["user_ordernum"])
        del all["ordertype_%s_ratio"%i]

    #所有的
    for c in ["orderbehavior_1_ratio","orderbehavior_2_ratio","orderbehavior_6_ratio","orderbehavior_7_ratio",
               #"user_roomservice_4_0ratio","user_roomservice_4_1ratio","user_roomservice_4_2ratio","user_roomservice_4_3ratio","user_roomservice_4_4ratio","user_roomservice_4_5ratio","user_roomservice_3_123ratio","user_roomservice_6_2ratio","user_roomservice_6_1ratio","user_roomservice_6_0ratio","user_roomservice_5_1ratio","user_roomservice_7_0ratio","user_roomservice_2_1ratio","user_roomservice_8_1ratio","user_roomservice_5_345ratio"

              ]:
        all[c]=map(lambda x,y:x*y,all[c],all["user_ordernum"])

    #一周的
    for c in ["orderbehavior_3_ratio_1week","orderbehavior_4_ratio_1week","orderbehavior_5_ratio_1week",
              #"user_roomservice_3_123ratio_1week","user_roomservice_7_1ratio_1week","user_roomservice_7_0ratio_1week","user_roomservice_4_5ratio_1week","user_roomservice_4_4ratio_1week","user_roomservice_4_2ratio_1week","user_roomservice_4_3ratio_1week","user_roomservice_4_0ratio_1week"
              ]:
        all[c] = map(lambda x,y: x * y, all[c], all["user_ordnum_1week"])

    #一个月的
    for c in ["orderbehavior_3_ratio_1month","orderbehavior_4_ratio_1month","orderbehavior_5_ratio_1month",
              #"user_roomservice_3_123ratio_1month", "user_roomservice_7_1ratio_1month", "user_roomservice_7_0ratio_1month","user_roomservice_4_5ratio_1month", "user_roomservice_4_4ratio_1month", "user_roomservice_4_2ratio_1month","user_roomservice_4_3ratio_1month", "user_roomservice_4_0ratio_1month"

              ]:
        all[c] = map(lambda x,y: x * y, all[c], all["user_ordnum_1month"])

    #三个月的
    for c in ["orderbehavior_3_ratio_3month","orderbehavior_4_ratio_3month","orderbehavior_5_ratio_3month",
              #"user_roomservice_3_123ratio_3month", "user_roomservice_7_1ratio_3month", "user_roomservice_7_0ratio_3month","user_roomservice_4_5ratio_3month", "user_roomservice_4_4ratio_3month", "user_roomservice_4_2ratio_3month","user_roomservice_4_3ratio_3month", "user_roomservice_4_0ratio_3month"

              ]:
        all[c] = map(lambda x,y: x * y, all[c], all["user_ordnum_3month"])


    all["price_star"]=all["price_deduct"]/(all["star"]-1)
    all["price_minarea"]=all["price_deduct"]/(all["basic_minarea"]-1)

    all["star_dif"]=all["user_avgstar"]-all["star"]

    all["price_ave_dif_rt"]=all["price_deduct"]/all["user_avgdealprice"]
    all["price_ave_star_dif"]=all["price_deduct"]/all["user_avgprice_star"]
    all["price_h_w_rt"]=all["user_avgdealpriceholiday"]/all["user_avgdealpriceworkday"]

    all["price_ave_dif"] = all["price_deduct"] - all["user_avgdealprice"]

    all["user_roomservice_4_32_rt"]=all["user_roomservice_4_3ratio"]/all["user_roomservice_4_2ratio"]
    all["user_roomservice_4_43_rt"]=all["user_roomservice_4_4ratio"]/all["user_roomservice_4_3ratio"]

    print all.shape

    online = model.predict(all.values)
    online = pd.DataFrame(online)
    online.columns = ["prob"]
    online["orderid"] = all["orderid"].values
    online["basicroomid"] = all["basicroomid"].values
    online["predict_roomid"] = all["roomid_ori"].values
    online["roomid"] = all["roomid"].values

    if j==0:
        result=online
    else:
        result=result.append(online)

result.to_csv("all_result_v29_24680_test_feature.csv",index=None)
del result["basicroomid"]
del result["roomid"]
result = result.sort_values("prob")
del result["prob"]
result = result.drop_duplicates("orderid", keep="last")
result["orderid"]=result["orderid"].apply(lambda x:"ORDER_"+str(x))
result["predict_roomid"]=result["predict_roomid"].apply(lambda x:"ROOM_"+str(x))
result.to_csv("sub_v29_24680.csv",index=None)


for j in [1,3,5,7,9]:
    print j
    all=pd.read_csv("try/offline_%s.csv"%j)
    del all["orderlabel"]

    # test中的basic_30days_ordnumratio数据异常，需要重新设置
    del all["basic_30days_ordnumratio"]
    all = all.merge(trick_basic_30d_fill, on=["hotelid", "basicroomid", "orderdate"], how="left")
    # 使用穿越特征
    all = all.merge(trick_1, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all = all.merge(trick_2, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all = all.merge(trick_3, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all["basic_trick_today"] = all["trick_1"] - all["basic_recent3_ordernum_ratio"]

    all = all.merge(trick_room, on=["hotelid", "roomid", "orderdate"], how="left")
    all["room_trick_today"] = all["trick_room"] - all["room_30days_ordnumratio"]

    all = all.merge(trick_basic_7d, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all["basic_7d_trick_today"] = all["trick_basic_7d"] - all["basic_week_ordernum_ratio"]

    all = all.merge(trick_basic_30d, on=["hotelid", "basicroomid", "orderdate"], how="left")
    all["basic_30d_trick_today"] = all["trick_basic_30d"] - all["basic_30days_ordnumratio"]

    all = all.merge(trick_basic_realratio, on=["basicroomid", "orderdate"], how="left")
    all["basic_30days_realratio_today"] = all["trick_basic_30days_realratio"] - all["basic_30days_realratio"]
    all = all.merge(trick_room_realratio, on=["roomid", "orderdate"], how="left")
    all["room_30days_realratio_today"] = all["trick_room_30days_realratio"] - all["room_30days_realratio"]


    all["roomid_ori"] = all["roomid"]
    all["roomid"] = map(lambda x, y: int(str(x)[:-len(str(y))]), all["roomid"], all["rank"])

    all = all.merge(basicroom_mean, on="basicroomid", how="left").fillna(0)
    all = all.merge(basicroom_sum, on="basicroomid", how="left").fillna(0)

    all=df_median(all)
    all=df_min(all)
    all=df_min_orderid(all)

    all["basicroomid_price_rank"] = all['price_deduct'].groupby([all['orderid'], all['basicroomid']]).rank()
    all["orderid_price_deduct_min_rank"] = all['orderid_price_deduct_min'].groupby(all['orderid']).rank()

    all = df_rank_mean(all)
    all = df_roomrank_mean(all)

    #添加新特征20170527
    #平均值
    all=merge_mean(all,["basicroomid"],"basic_week_ordernum_ratio","basic_week_ordernum_ratio_mean")
    all=merge_mean(all,["basicroomid"],"basic_recent3_ordernum_ratio","basic_recent3_ordernum_ratio_mean")
    all=merge_mean(all,["basicroomid"],"basic_comment_ratio","basic_comment_ratio_mean")
    all=merge_mean(all,["basicroomid"],"basic_30days_ordnumratio","basic_30days_ordnumratio_mean")
    all=merge_mean(all,["basicroomid"],"basic_30days_realratio","basic_30days_realratio_mean")
    all=merge_mean(all,["roomid"],"room_30days_ordnumratio","room_30days_ordnumratio_mean")
    all=merge_mean(all,["roomid"],"room_30days_realratio","room_30days_realratio_mean")

    all["city_num"]=all["user_ordernum"]/all["user_citynum"]
    all["area_price"]=all["user_avgprice"]/all["user_avgroomarea"]
    all["price_max_min_rt"]=all["user_maxprice"]/all["user_minprice"]
    all["basicroomid_price_deduct_min_minprice_rt"]=all["basicroomid_price_deduct_min"]/all["user_minprice"]

    all["price_dif"]=all["basicroomid_price_deduct_min"]-all["price_deduct"]
    all["price_dif_hotel"]=all["basicroomid_price_deduct_min"]-all["hotel_minprice_lastord"]
    all["price_dif_basic"]=all["basicroomid_price_deduct_min"]-all["basic_minprice_lastord"]

    all["price_dif_rt"]=all["basicroomid_price_deduct_min"]/all["price_deduct"]
    all["price_dif_hotel_rt"]=all["basicroomid_price_deduct_min"]/all["hotel_minprice_lastord"]
    all["price_dif_basic_rt"]=all["basicroomid_price_deduct_min"]/all["basic_minprice_lastord"]

    all["price_dif_hotel"]=all["orderid_price_deduct_min"]-all["price_deduct"]
    all["price_dif_hotel_hotel"]=all["orderid_price_deduct_min"]-all["hotel_minprice_lastord"]
    all["price_dif_basic_hotel"]=all["orderid_price_deduct_min"]-all["basic_minprice_lastord"]

    all["price_dif_hotel_rt"]=all["orderid_price_deduct_min"]/all["price_deduct"]
    all["price_dif_hotel_hotel_rt"]=all["orderid_price_deduct_min"]/all["hotel_minprice_lastord"]
    all["price_dif_basic_hotel_rt"]=all["orderid_price_deduct_min"]/all["basic_minprice_lastord"]

    #all["order_basic_minprice_dif"]=all["basicroomid_price_deduct_min"]-all["orderid_price_deduct_min"]
    all["order_basic_minprice_rt"]=all["basicroomid_price_deduct_min"]/all["orderid_price_deduct_min"]
    #all["hotel_basic_minprice_lastord_rt"]=all["basic_minprice_lastord"]/all["hotel_minprice_lastord"]

    #上次订购的价格和当时最低价的比
    all["hotel_last_price_min_rt"]=all["price_last_lastord"]/all["hotel_minprice_lastord"]
    all["basic_last_price_min_rt"]=all["price_last_lastord"]/all["basic_minprice_lastord"]
    all["hotel_last_price_min_dif"]=all["price_last_lastord"]-all["hotel_minprice_lastord"]
    all["basic_last_price_min_dif"]=all["price_last_lastord"]-all["basic_minprice_lastord"]


    all["price_tail1"]=all["price_deduct"]%10
    all["price_tail1"]=map(lambda x:1 if x==4 or x==7 else 0,all["price_tail1"])
    #all["price_tail2"]=all["price_deduct"]%100
    all["basic_equal"]=map(lambda x,y:1 if x==y else 0,all["basicroomid"],all["basicroomid_lastord"])
    #del all["basicroomid_lastord"]
    all["room_equal"]=map(lambda x,y:1 if x==y else 0,all["roomid"],all["roomid_lastord"])
    #del all["roomid_lastord"]
    all["hotel_equal"]=map(lambda x,y:1 if x==y else 0,all["hotelid"],all["hotelid_lastord"])
    #del all["hotelid_lastord"]
    all["rank_equal"]=map(lambda x,y:1 if x==y else 0,all["rank"],all["rank_lastord"])

    #价格高低
    all["price_dx"] = map(lambda x, y: x-y, all["price_deduct"], all["price_last_lastord"])

    all["return_dx"] = map(lambda x, y: x-y, all["returnvalue"], all["return_lastord"])

    all["price_ori"] = map(lambda x, y:x+y, all["price_deduct"], all["returnvalue"])


    for i in [2,3,4,5,6,8]:
        all["service_equal_%s"%i] = map(lambda x, y: 1 if x == y else 0, all["roomservice_%s"%i], all["roomservice_%s_lastord"%i])
        del all["roomservice_%s_lastord"%i]

    for i in [2,3,4,5,6]:
        all["roomtag_equal_%s"%i] = map(lambda x, y: 1 if x == y else 0, all["roomtag_%s"%i], all["roomtag_%s_lastord"%i])
        del all["roomtag_%s_lastord"%i]

    for i in [1,2,3,4,5,6,7,8,9,10,11]:
        all["ordertype_%s_num"%i] = map(lambda x, y:x*y, all["ordertype_%s_ratio"%i], all["user_ordernum"])
        del all["ordertype_%s_ratio"%i]

    #所有的
    for c in ["orderbehavior_1_ratio","orderbehavior_2_ratio","orderbehavior_6_ratio","orderbehavior_7_ratio",
               #"user_roomservice_4_0ratio","user_roomservice_4_1ratio","user_roomservice_4_2ratio","user_roomservice_4_3ratio","user_roomservice_4_4ratio","user_roomservice_4_5ratio","user_roomservice_3_123ratio","user_roomservice_6_2ratio","user_roomservice_6_1ratio","user_roomservice_6_0ratio","user_roomservice_5_1ratio","user_roomservice_7_0ratio","user_roomservice_2_1ratio","user_roomservice_8_1ratio","user_roomservice_5_345ratio"

              ]:
        all[c]=map(lambda x,y:x*y,all[c],all["user_ordernum"])

    #一周的
    for c in ["orderbehavior_3_ratio_1week","orderbehavior_4_ratio_1week","orderbehavior_5_ratio_1week",
              #"user_roomservice_3_123ratio_1week","user_roomservice_7_1ratio_1week","user_roomservice_7_0ratio_1week","user_roomservice_4_5ratio_1week","user_roomservice_4_4ratio_1week","user_roomservice_4_2ratio_1week","user_roomservice_4_3ratio_1week","user_roomservice_4_0ratio_1week"
              ]:
        all[c] = map(lambda x,y: x * y, all[c], all["user_ordnum_1week"])

    #一个月的
    for c in ["orderbehavior_3_ratio_1month","orderbehavior_4_ratio_1month","orderbehavior_5_ratio_1month",
              #"user_roomservice_3_123ratio_1month", "user_roomservice_7_1ratio_1month", "user_roomservice_7_0ratio_1month","user_roomservice_4_5ratio_1month", "user_roomservice_4_4ratio_1month", "user_roomservice_4_2ratio_1month","user_roomservice_4_3ratio_1month", "user_roomservice_4_0ratio_1month"

              ]:
        all[c] = map(lambda x,y: x * y, all[c], all["user_ordnum_1month"])

    #三个月的
    for c in ["orderbehavior_3_ratio_3month","orderbehavior_4_ratio_3month","orderbehavior_5_ratio_3month",
              #"user_roomservice_3_123ratio_3month", "user_roomservice_7_1ratio_3month", "user_roomservice_7_0ratio_3month","user_roomservice_4_5ratio_3month", "user_roomservice_4_4ratio_3month", "user_roomservice_4_2ratio_3month","user_roomservice_4_3ratio_3month", "user_roomservice_4_0ratio_3month"

              ]:
        all[c] = map(lambda x,y: x * y, all[c], all["user_ordnum_3month"])


    all["price_star"]=all["price_deduct"]/(all["star"]-1)
    all["price_minarea"]=all["price_deduct"]/(all["basic_minarea"]-1)

    all["star_dif"]=all["user_avgstar"]-all["star"]

    all["price_ave_dif_rt"]=all["price_deduct"]/all["user_avgdealprice"]
    all["price_ave_star_dif"]=all["price_deduct"]/all["user_avgprice_star"]
    all["price_h_w_rt"]=all["user_avgdealpriceholiday"]/all["user_avgdealpriceworkday"]

    all["price_ave_dif"] = all["price_deduct"] - all["user_avgdealprice"]

    all["user_roomservice_4_32_rt"]=all["user_roomservice_4_3ratio"]/all["user_roomservice_4_2ratio"]
    all["user_roomservice_4_43_rt"]=all["user_roomservice_4_4ratio"]/all["user_roomservice_4_3ratio"]

    print all.shape

    online = model.predict(all.values)
    online = pd.DataFrame(online)
    online.columns = ["prob"]
    online["orderid"] = all["orderid"].values
    online["basicroomid"] = all["basicroomid"].values
    online["predict_roomid"] = all["roomid_ori"].values
    online["roomid"] = all["roomid"].values

    if j==0 or j==1:
        result=online
    else:
        result=result.append(online)

result.to_csv("all_result_v29_24680_train_feature.csv",index=None)

