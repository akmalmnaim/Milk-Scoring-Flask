# from keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import pickle
import math, os

df = pd.read_csv("milknew.csv", )
Target = {k: v for k, v in zip(['high', 'low', 'medium'], list(range(3)))}
for i in range(df.shape[0]):
    df.iloc[i, -1] = Target[df.iloc[i, -1]]


X = np.array(df.iloc[:, 0:-1])
y = np.asarray(df.iloc[:, -1]).astype('int64')

X_train, X_test, y_train, y_test = train_test_split(X, y)
sc = MinMaxScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


# def GELU(x):
#     res = 0.5 * x * (1 + tf.nn.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * (x ** 3))))
#     return res

# class ResMLPBlock(tf.keras.layers.Layer):
#     def __init__(self, units, residual_path):
#         super(ResMLPBlock, self).__init__()
#         self.residual_path = residual_path
#         self.D1 = Dense(units, activation='relu')
#         self.D2 = Dense(units, activation='relu')

#         if self.residual_path:
#             self.D3 = Dense(units)
#             self.D4 = Dense(units)

#     def call(self, inputs):
#         residual = inputs

#         x = self.D1(inputs)
#         y = self.D2(x)

#         if self.residual_path:
#             residual = self.D3(inputs)
#             residual = GELU(residual)
#             residual = self.D4(residual)
#             residual = GELU(residual)

#         output = y + residual
#         return output


# # ResMLP网络结构
# class ResMLP(tf.keras.Model):
#     def __init__(self, initial_filters, block_list, num_classes):
#         super(ResMLP, self).__init__()
#         self.initial_filters = initial_filters
#         self.block_list = block_list

#         self.D1 = Dense(self.initial_filters, activation='relu')
#         self.B1 = BatchNormalization()

#         self.blocks = tf.keras.models.Sequential()
#         for block_id in range(len(block_list)):
#             for layer_id in range(block_list[block_id]):
#                 if block_id != 0 and layer_id == 0:
#                     block = ResMLPBlock(units=self.initial_filters, residual_path=True)
#                 else:
#                     block = ResMLPBlock(units=self.initial_filters, residual_path=False)
#                 self.blocks.add(block)
#             self.initial_filters *= 2

#         self.D2 = Dense(num_classes, activation='softmax')


#     def call(self, inputs):
#         x = self.D1(inputs)
#         x = self.B1(x)
#         x = self.blocks(x)
#         y = self.D2(x)
#         return y



# net = ResMLP(initial_filters=32, block_list=[2, 2, 2], num_classes=3)

# net.compile(optimizer='adam',
#             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#             metrics=['sparse_categorical_accuracy'])

# history = net.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))


# net.save_weights('model1.hdf5')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report

# clf = AdaBoostClassifier(n_estimators=100, random_state=0)
# clf.fit(X_train, y_train)

model=pickle.load(open('modeladaboost2.pkl','rb'))


int_feature=(6.6,35,1,0,1,0,254)
reshaped=[np.array(int_feature)]
std_data = sc.transform(reshaped)
print(int_feature)
print(std_data)
prediction=model.predict(std_data)[0]
print(prediction)
