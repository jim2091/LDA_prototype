# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 21:19:39 2021

@author: LJB
"""

import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
plt.rc('font', family='Malgun Gothic')


#########################데이터 로드 및 전처리#####################################

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
#min_face_per_person : 인물당 최소 사진의 수. resize : 원본보다 일정 비율로 크기를 축소
image_shape = people.images[0].shape

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
# np.unique : 배열 내의 중복된 원소 제거 후 유일한 원소 반환.
# 근데 이 마스크랑 포문 부분이 왜 필요한지 잘 모르겠어요
# 이걸 통해 만들어진건 모든 원소가 1로 구성된 마스크고 그게 쓰이는게 바로 아랫부분인데
# 저걸 [mask]를 빼도 똑같은 결과가 들어가요
# 이건 찾아봐도 잘 모르겠습니다!

X_people = people.data[mask]
y_people = people.target[mask]

# 0~255 사이의 이미지의 픽셀 값을 0~1 사이로 스케일 조정합니다.
# (옮긴이) MinMaxScaler를 적용하는 것과 거의 동일합니다.
X_people = X_people / 255.

X_train, X_test, y_train, y_test = \
train_test_split(X_people, y_people, stratify=y_people, random_state=0)


###3개의 얼굴들을 다양한 개수의 특성으로 분해, 재구성한 결과
mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)

###이번엔 15개의 특성으로 나눔
nmf = NMF(n_components=15, random_state=0, max_iter=1000, tol=1e-2)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)
# 각각에 들어있는건 train, test셋 각각의 이미지 별 특성 적용 가중치(X=WH에서 H부분)


###15개의 특성들을 확인 가능.
fig, axes = plt.subplots(3, 5, figsize=(15, 12),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title("component {}".format(i))
    
### 4, 5 얼굴이 각각 오른쪽, 왼쪽을 보고있는 이미지임을 확인.
compn = 4
# 4번째 성분으로 정렬하여 처음 10개 이미지를 출력합니다
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))
    
compn = 5
# 5번째 성분으로 정렬하여 처음 10개 이미지를 출력합니다
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))    
