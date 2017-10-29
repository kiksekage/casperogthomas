fp = "../testkode/"

pred_dict = {0 : 'anger-pred.txt', 1 : 'fear-pred.txt', 2 : 'joy-pred.txt', 3 : 'sadness-pred.txt'}
pred_file = "../testkode/preds.txt"
pred_fold = fp + "preds/"

fp17 = fp + "data17/"
fp18 = fp + "data18/"

fp17_train = fp17 + "train/"
train_class_17 = [fp17_train + "anger-ratings-0to1.train.txt", fp17_train + "fear-ratings-0to1.train.txt",
            fp17_train + "joy-ratings-0to1.train.txt", fp17_train + "sadness-ratings-0to1.train.txt"]

fp17_test = fp17 + "test/"
test_class_17 = [fp17_test + "anger-ratings-0to1.test.target.txt", fp17_test + "fear-ratings-0to1.test.target.txt",
           fp17_test + "joy-ratings-0to1.test.target.txt", fp17_test + "sadness-ratings-0to1.test.target.txt"]

fp17_dev = fp17 + "dev/"
dev_class_17 = [fp17_dev + "anger-ratings-0to1.dev.txt", fp17_dev + "fear-ratings-0to1.dev.txt",
          fp17_dev + "joy-ratings-0to1.dev.txt", fp17_dev + "sadness-ratings-0to1.dev.txt"]

########################### ENGLISH DATASETS, REG ###########################

fp18_reg_train_en = fp18 + "reg/2018-EI-reg-En-train/"
train_reg_18_en = [fp18_reg_train_en + "2018-EI-reg-En-anger-train.txt", fp18_reg_train_en + "2018-EI-reg-En-fear-train.txt",
                 fp18_reg_train_en + "2018-EI-reg-En-joy-train.txt", fp18_reg_train_en + "2018-EI-reg-En-sadness-train.txt"]

fp18_reg_dev_en = fp18 + "reg/2018-EI-reg-En-dev/"
dev_reg_18_en = [fp18_reg_dev_en + "2018-EI-reg-En-anger-dev.txt", fp18_reg_dev_en + "2018-EI-reg-En-fear-dev.txt",
                 fp18_reg_dev_en + "2018-EI-reg-En-joy-dev.txt", fp18_reg_dev_en + "2018-EI-reg-En-sadness-dev.txt"]

########################### ARABIC DATASETS, REG ###########################

fp18_reg_train_ar = fp18 + "reg/2018-EI-reg-Ar-train/"
train_reg_18_ar = [fp18_reg_train_ar + "2018-EI-reg-Ar-anger-train.txt", fp18_reg_train_ar + "2018-EI-reg-Ar-fear-train.txt",
                 fp18_reg_train_ar + "2018-EI-reg-Ar-joy-train.txt", fp18_reg_train_ar + "2018-EI-reg-Ar-sadness-train.txt"]

fp18_reg_dev_ar = fp18 + "reg/2018-EI-reg-Ar-dev/"
dev_reg_18_ar = [fp18_reg_dev_ar + "2018-EI-reg-Ar-anger-dev.txt", fp18_reg_dev_ar + "2018-EI-reg-Ar-fear-dev.txt",
                 fp18_reg_dev_ar + "2018-EI-reg-Ar-joy-dev.txt", fp18_reg_dev_ar + "2018-EI-reg-Ar-sadness-dev.txt"]

########################### SPANISH DATASETS, REG ###########################

fp18_reg_train_es = fp18 + "reg/2018-EI-reg-Es-train/"
train_reg_18_es = [fp18_reg_train_es + "2018-EI-reg-Es-anger-train.txt", fp18_reg_train_es + "2018-EI-reg-Es-fear-train.txt",
                 fp18_reg_train_es + "2018-EI-reg-Es-joy-train.txt", fp18_reg_train_es + "2018-EI-reg-Es-sadness-train.txt"]

fp18_reg_dev_es = fp18 + "reg/2018-EI-reg-Es-dev/"
dev_reg_18_es = [fp18_reg_dev_es + "2018-EI-reg-Es-anger-dev.txt", fp18_reg_dev_es + "2018-EI-reg-Es-fear-dev.txt",
                 fp18_reg_dev_es + "2018-EI-reg-Es-joy-dev.txt", fp18_reg_dev_es + "2018-EI-reg-Es-sadness-dev.txt"]

########################### ENGLISH DATASETS, CLASS ###########################

fp18_class_train_en = fp18 + "class/2018-EI-oc-En-train/"
train_class_18_en = [fp18_class_train_en + "2018-EI-oc-En-anger-train.txt", fp18_class_train_en + "2018-EI-oc-En-fear-train.txt",
                 fp18_class_train_en + "2018-EI-oc-En-joy-train.txt", fp18_class_train_en + "2018-EI-oc-En-sadness-train.txt"]

fp18_class_dev_en = fp18 + "class/2018-EI-oc-En-dev/"
dev_class_18_en = [fp18_class_dev_en + "2018-EI-oc-En-anger-dev.txt", fp18_class_dev_en + "2018-EI-oc-En-fear-dev.txt",
                 fp18_class_dev_en + "2018-EI-oc-En-joy-dev.txt", fp18_class_dev_en + "2018-EI-oc-En-sadness-dev.txt"]

########################### ARABIC DATASETS, CLASS ###########################

fp18_class_train_ar = fp18 + "class/2018-EI-oc-Ar-train/"
train_class_18_ar = [fp18_class_train_ar + "2018-EI-oc-Ar-anger-train.txt", fp18_class_train_ar + "2018-EI-oc-Ar-fear-train.txt",
                 fp18_class_train_ar + "2018-EI-oc-Ar-joy-train.txt", fp18_class_train_ar + "2018-EI-oc-Ar-sadness-train.txt"]

fp18_class_dev_ar = fp18 + "class/2018-EI-oc-Ar-dev/"
dev_class_18_ar = [fp18_class_dev_ar + "2018-EI-oc-Ar-anger-dev.txt", fp18_class_dev_ar + "2018-EI-oc-Ar-fear-dev.txt",
                 fp18_class_dev_ar + "2018-EI-oc-Ar-joy-dev.txt", fp18_class_dev_ar + "2018-EI-oc-Ar-sadness-dev.txt"]

########################### SPANISH DATASETS, CLASS ###########################

fp18_class_train_es = fp18 + "class/2018-EI-oc-Es-train/"
train_class_18_es = [fp18_class_train_es + "2018-EI-oc-Es-anger-train.txt", fp18_class_train_es + "2018-EI-oc-Es-fear-train.txt",
                 fp18_class_train_es + "2018-EI-oc-Es-joy-train.txt", fp18_class_train_es + "2018-EI-oc-Es-sadness-train.txt"]

fp18_class_dev_es = fp18 + "class/2018-EI-oc-Es-dev/"
dev_class_18_es = [fp18_class_dev_es + "2018-EI-oc-Es-anger-dev.txt", fp18_class_dev_es + "2018-EI-oc-Es-fear-dev.txt",
                 fp18_class_dev_es + "2018-EI-oc-Es-joy-dev.txt", fp18_class_dev_es + "2018-EI-oc-Es-sadness-dev.txt"]

########################### FILE LIST, 17 ###########################

file_list_17 = [train_class_17, test_class_17, dev_class_17]

list_dict_17 = {"train" : 0, "test" : 1, "dev" : 2}

########################### FILE LIST, 18 ###########################

file_list_18 = [train_reg_18_en, dev_reg_18_en, train_reg_18_ar, dev_reg_18_ar, train_reg_18_es, dev_reg_18_es,
            train_class_18_en, dev_class_18_en, train_class_18_ar, dev_class_18_ar, train_class_18_es, dev_class_18_es]

list_dict_18 = {"reg en train" : 0, "reg en dev" : 1, "reg ar train" : 2, "reg ar dev" : 3, "reg es train" : 4, "reg es dev" : 5,
                "class en train" : 6, "class en dev" : 7, "class ar train" : 8, "class ar dev" : 9, "class es train" : 10, "class es dev" : 11}

def filepath_returner(task, language, year):
    if year == '18':
        dict_string_train = ' '.join([task, language, "train"])
        dict_string_dev = ' '.join([task, language, "dev"])
        return file_list_18[list_dict_18[dict_string_train]], file_list_18[list_dict_18[dict_string_dev]]
    elif year == '17':
        return file_list_17[list_dict_17["train"]], file_list_17[list_dict_17["dev"]]

#print(filepath_returner("en", "17", "class", "train"))
#print(filepath_returner("class", "en", "18"))