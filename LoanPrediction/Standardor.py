import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm


import Utility

def load_user_info(path):
     data_file = open(path)
     # key is user id, value is user info vector
     user_info_dict = {}
     first_line_flag = True
     for current_line in data_file:
        if first_line_flag:
            first_line_flag = False
            continue

        fields = current_line.split(',')
        user_id = fields[0]
        age = fields[1]
        if user_info_dict.has_key(user_id):
            original_vector = user_info_dict[user_id]
            if (original_vector[0] == 'NONE'):
                original_vector[0] = age
            for o_index in range(1, len(original_vector)):
                if original_vector[o_index] == '0' or \
                                original_vector[o_index] == '0.0'or \
                                original_vector[o_index] == 'NA' or \
                                len(original_vector[o_index]) == 0:
                    original_vector[o_index] = fields[o_index + 1]
        else:
            del fields[0]
            user_info_dict[user_id] = fields
     data_file.close()
     return user_info_dict

def load_target_info(path):
    user_second_loan = {}
    file = open(path)
    first_line_flag = True
    for current_line in file:
        if first_line_flag:
            first_line_flag = False
            continue

        fields = current_line.split(',')
        user_second_loan[fields[0]] = int(fields[1])
    file.close()
    return user_second_loan

def save_filtered_user_info(user_info_dict, file_path):
    file = open(file_path, 'w')
    for key, vector in user_info_dict.items():
        file.write(key + ',')
        for i in range(0, len(vector) - 1):
            file.write(vector[i] + ',')
        file.write(vector[-1] + '\n')

def standarize_user_info(path):
    user_info_dict = load_user_info(path)
    print user_info_dict
    # convert string to number
    #key vector index ; value [missing number, total number]
    statistic = {}
    for key, vector in user_info_dict.items():
        for cur_index in range(0, len(vector)):
            if not str.isdigit(vector[cur_index]) or float(vector[cur_index]) == 0.0:
                if (statistic.has_key(cur_index)):
                    statistic[cur_index][0] += 1
                    statistic[cur_index][1] += 1
                else:
                    statistic[cur_index] = [1, 1]
            else:
                if (statistic.has_key(cur_index)):
                    statistic[cur_index][1] += 1
                else:
                    statistic[cur_index] = [0, 1]
    print statistic

# for continuous value, '' and '0' and 'NA' are all set to 'NA'
def seperete_continuous_and_discrete_features(user_info_dict):
    continuous_feature_index = [0, 2, 3, 11, 13, 14, 16, 17]
    user_continuous_info = {}
    user_discrete_info = {}
    discrete_feature_encoding_schema = {} #key is column index of original vector, value is possible features for this column
    for user_id, original_vec in user_info_dict.items():
        user_continuous_info[user_id] = []
        user_discrete_info[user_id] = []
        for feature_index in range(0, len(original_vec)):
            current_feature_value = original_vec[feature_index]
            if feature_index in continuous_feature_index: # it is continuous feature
                if not str.isdigit(current_feature_value) or float(current_feature_value) == 0.0:
                    user_continuous_info[user_id].append(np.nan)
                else:
                    user_continuous_info[user_id].append(float(current_feature_value))
            else: # it is discrete feature
                if current_feature_value == 'NA':
                    user_discrete_info[user_id].append(0)
                elif current_feature_value == '':
                    user_discrete_info[user_id].append(99) # -1 means ''
                else:
                    user_discrete_info[user_id].append(int(current_feature_value))
                #if discrete_feature_encoding_schema.has_key(feature_index):
                #    if current_feature_value not in discrete_feature_encoding_schema[feature_index]:
                #        discrete_feature_encoding_schema[feature_index].append(current_feature_value)
                #else:
                #    discrete_feature_encoding_schema[feature_index] = [current_feature_value]
    return user_discrete_info, user_continuous_info

def filling_missing_value(user_info_dict):
    user_discrete_info, user_continuous_info = seperete_continuous_and_discrete_features(user_info_dict)
    # filling continous features
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    user_continuous_info_matrix = Utility.dict2Array(user_continuous_info)
    imp.fit(user_continuous_info_matrix)
    full_user_continuous_info_matrix = imp.transform(user_continuous_info_matrix)

    # encode discrete info
    enc = preprocessing.OneHotEncoder()
    user_discrete_info_matrix = Utility.dict2Array(user_discrete_info)
    enc.fit(user_discrete_info_matrix)
    full_user_discrete_info_matrix = enc.transform(user_discrete_info_matrix).toarray()

    return full_user_discrete_info_matrix, full_user_continuous_info_matrix, imp, enc
#
user_info_dict = load_user_info('../new_user_info.csv')
user_second_loan = load_target_info('../train.txt')

test_user_info_dict = {}
train_user_info_dict = {}
train_target_dict = {}

for user_id in user_info_dict.keys():
    if user_second_loan.has_key(user_id): # it is for training
        train_target_dict[user_id] = user_second_loan[user_id]
        train_user_info_dict[user_id] = user_info_dict[user_id]
    else: # it is for testing
        test_user_info_dict[user_id] = user_info_dict[user_id]

full_user_discrete_info_matrix, full_user_continuous_info_matrix, imp, enc = filling_missing_value(train_user_info_dict)

min_max_scaler = preprocessing.MinMaxScaler()
full_user_continuous_info_matrix = min_max_scaler.fit_transform(np.array(full_user_continuous_info_matrix))

train_data = np.concatenate((np.array(full_user_discrete_info_matrix), np.array(full_user_continuous_info_matrix)), axis=1)
train_target = Utility.dict2Array(train_target_dict)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data, train_target, test_size=0.3, random_state=0)

from sklearn.naive_bayes import BernoulliNB
gnb = BernoulliNB ()
clf = gnb.fit(X_train, y_train)
print clf.predict(X_test)
print clf.predict_proba(X_test)
print y_test

#from sklearn import tree
#clf = tree.DecisionTreeRegressor().fit(X_train, y_train)
#print clf.score(X_test, y_test)

#from sklearn.neighbors import KNeighborsRegressor
#clf = KNeighborsRegressor(n_neighbors=10).fit(X_train, y_train)
#print clf.score(X_test, y_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis().fit(X_train, y_train)
print clf.score(X_test, y_test)
#user_info_dict = load_user_info('../user_info.txt')
#save_filtered_user_info(user_info_dict, '../new_user_info.csv')
#standarize_user_info('../data/user_info.txt')

