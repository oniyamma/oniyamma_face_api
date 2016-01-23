#!/usr/bin/env python
# -*- coding: UTF-8 -*-

 
import math
import os
import sys
import csv
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
import numpy as np

class Classify_face:
    def load_csv(self, file_name):
        f = open(file_name, 'rb')  
        dataReader = csv.reader(f)

        points = []
        count = 0
        for row in dataReader:
            print row
            points.append((float(row[1]), float(row[2]), float(row[3])))

        return points

    def load_csv2(self, file_name):
        f = open(file_name, 'rb')  
        dataReader = csv.reader(f)

        face = []
        label = []
        points = []
        for x, row in enumerate(dataReader):
            if (x % 33 == 0) :
                label.append(int(row[0]))

            else:
                #print row
                #print x
                points.append((float(row[1]), float(row[2]), float(row[3])))

                if x % 33 == 32:
                    face.append(points[:])
                    points = [] 

        return face, label

    def load_string(self, str):
        print str
        f = open('temp', 'w')
        f.write(str)
        f.close()

        return self.load_csv('temp')

    def distance(self,point1, point2):
        x2 = math.pow( (point1[0] -point2[0]), 2 )
        y2 = math.pow( (point1[1] -point2[1]), 2 )
        return math.sqrt(x2 + y2)

    def distance_pattern(self,points):
        eye_distance = self.distance(points[0], points[1])
        orig_point = points[0]

        dists = []
        # スコア計算
        for point in points:
            dists.append( self.distance(orig_point, point) / eye_distance )

        return dists

    def match_score(self,dists1, dists2):
        sum = 0.0
        for dist1, dist2 in zip(dists1, dists2):
            sum += math.fabs(dist1 - dist2)

        return sum

    def laern_and_classify(self, dataset_face, dataset_label, eval_face):
        dist = []
        for f in dataset_face:
            dist.append(self.distance_pattern(f))

        data_train, data_test, label_train, label_test = train_test_split(dist, dataset_label)
        #print label_train 

        estimator = LinearSVC(C=1.0)

        estimator.fit(data_train, label_train)

        # テストデータを分類する
        #res =  estimator.predict(data_test)
        #print res
        #print np.asarray(label_test)
    
        # 分類
        res = estimator.predict([self.distance_pattern(eval_face)])
        print res[0] 

if __name__ == '__main__':
     
   # orig_name = sys.argv[1]
   # target_name = sys.argv[2]
   # orig_points = load_csv(orig_name)
   # target_points = load_csv(target_name)
    
   # score = match_score(distance_pattern(orig_points), distance_pattern(target_points))    
   # print score

    classifier = Classify_face()
    face, label = classifier.load_csv2(sys.argv[1])
    eval = classifier.load_string("".join(sys.stdin.readlines()))
    #eval = classifier.load_csv(sys.argv[2])
    classifier.laern_and_classify(face, label, eval)
    
    # デバッグ　ポイント
    # for point in points:
#        print("%.6f , %.6f ,%.6f" %(point[0], point[1], point[2])  )
    
