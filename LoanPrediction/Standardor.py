from sklearn import preprocessing

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
                if original_vector[o_index] == '0' or original_vector[o_index] == '0.0'or len(original_vector[o_index]) == 0:
                    original_vector[o_index] = fields[o_index + 1]
        else:
            del fields[0]
            del fields[-1]
            user_info_dict[user_id] = fields
     data_file.close()
     return user_info_dict

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

standarize_user_info('../data/user_info.txt')