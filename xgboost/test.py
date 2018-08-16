#coding:utf-8
'''
*******************************************************************************
*       Filename:  template.py
*    Description:  py file
*       
*        Version:  1.0
*        Created:  2018-08-11
*         Author:  chencheng
*
*        History:  initial draft
* 
* https://blog.csdn.net/c2a2o2/article/details/77646025
* https://zhuanlan.zhihu.com/p/28739256
* kaggle: https://www.kaggle.com/c/titanic
*******************************************************************************
'''

import numpy as np
import pandas as pd
import xgboost
import sklearn.model_selection


def ReadData():
	train_file = "/home/chen/dataset/kaggle/titanic/train.csv"
	test_file  = "/home/chen/dataset/kaggle/titanic/test.csv"
	# train = pd.read_csv(train_file, index_col="PassengerId")
	train = pd.read_csv(train_file)
	test  = pd.read_csv(test_file)
	return train, test


def CheckNaN(df):
    flag = df.isnull().any()
    #print(type(flag))
    #print(flag)
    if True in flag:
        return False
    else:
        return True


def CleanData(titanic):
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic["child"] = titanic["Age"].apply(lambda x: 1 if x < 15 else 0)

    titanic["sex"] = titanic["Sex"].apply(lambda x: 1 if x == "male" else 0)

    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    def getEmbark(Embarked):
        if Embarked == "S":
            return 1
        elif Embarked == "C":
            return 2
        else:
            return 3
    titanic["embark"] = titanic["Embarked"].apply(getEmbark)

    titanic["fimalysize"] = titanic["SibSp"] + titanic["Parch"] + 1

    # cabin
    def getCabin(cabin):
        if cabin == "N":
            return 0
        else:
            return 1
    titanic["cabin"] = titanic["Cabin"].apply(getCabin)

    # name
    def getName(name):
        if "Mrs" in str(name):
            return 1
        elif "Mr" in str(name):
            return 2
        else:
            return 0
    titanic["name"] = titanic["Name"].apply(getName)

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic


def main():
    ''' read data '''
    train_data, test_data = ReadData()
    '''
    print(train_data.head())
    print(train_data["Name"][1])
    print(test_data["Name"][0])
    print(type(test_data["PassengerId"]))
    '''

    ''' clean data '''
    train_data = CleanData(train_data)
    test_data  = CleanData(test_data)

    if (CheckNaN(train_data)):
        print("ok!")
    else:
        print("not ok!")


    ''' feature engineering '''
    features = ["Pclass", "sex", "child", "fimalysize", "Fare", "embark", "cabin"]

    ''' build model '''
    #clf = xgboost.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100,
    #                            silent=False, objective='binary:logistic')
    clf = xgboost.XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=32,
                                silent=True, objective='binary:logistic')

    clf.fit(train_data[features], train_data["Survived"])

    predictions = clf.predict(test_data[features])

    
    submission = pd.DataFrame({'PassengerId': test_data['PassengerId'],
                                'Survived': predictions})

    output_file_name = "gender_submission.csv"
    submission.to_csv(output_file_name, index=False)
    print("# %s generated!" %(output_file_name))

    print('this is xgboost!')


if __name__ == "__main__":
	main()