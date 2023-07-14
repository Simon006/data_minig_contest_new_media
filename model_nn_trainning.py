import torch
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation
from untils import read_events_heat
from process_1 import load_data_from_xlrx
cols_to_cluster = ["上海","云南","其他","内蒙古","北京", "台湾","吉林","四川","天津","宁夏","安徽",	"山东","山西",
                   "广东","广西","新疆","江苏","江西","河北","河南","浙江","海南","海外","湖北","湖南","澳门","甘肃","福建","西藏","贵州",
                   "辽宁","重庆","陕西","青海","香港","黑龙江","点赞-mean","点赞-max","点赞-var","点赞-sum","转发-mean","转发-max","转发-var",
                   "转发-sum","评论-mean","评论-max","评论-var","评论-sum","粉丝数-mean","粉丝数-max","粉丝数-var","粉丝数-sum","关注数-mean",
                   "关注数-max","关注数-var","关注数-sum","时间衰减系数"]

filename = "meta_data_v1.xlsx"
X_train, X_test, y_train, y_test = load_data_from_xlrx(filename=filename,cols_to_cluster=cols_to_cluster)

y_train_1 = y_train/60
y_test_1 = y_test/60

print("X_train.shape:",X_train.shape)
print("Y_train.shape:",X_test.shape)
model = Sequential([
    	Dense(64, input_shape=(57,)),
    	Activation('relu'),
        Dense(32),
        Activation('relu'),
   	    Dense(10),
        Activation('relu'),
        Dense(1),   
    	Activation('softmax'),])
 

opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_squared_error", optimizer=opt)
# train the model
print("[INFO] training model...")
model.fit(x=X_train, y=y_train_1, 
	validation_data=(X_test, y_test_1),
	epochs=50,verbose=2, batch_size=64)




