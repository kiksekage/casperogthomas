filename_dict = {'anger' : 0, 'fear' : 1, 'joy' : 2, 'sadness' : 3}

pred = "../testkode/preds.txt"
    
fp = "../testkode/"
fp17 = fp + "data17/"
fp18 = fp + "data18/"

preds = fp + "preds/"

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
dev_reg_18_en = [fp18_reg_dev_en + "2018-EI-reg-En-anger-train.txt", fp18_reg_dev_en + "2018-EI-reg-En-fear-train.txt",
                 fp18_reg_dev_en + "2018-EI-reg-En-joy-train.txt", fp18_reg_dev_en + "2018-EI-reg-En-sadness-train.txt"]

########################### ARABIC DATASETS, REG ###########################

fp18_reg_train_ar = fp18 + "reg/2018-EI-reg-Ar-train/"
train_reg_18_ar = [fp18_reg_train_ar + "2018-EI-reg-Ar-anger-train.txt", fp18_reg_train_ar + "2018-EI-reg-Ar-fear-train.txt",
                 fp18_reg_train_ar + "2018-EI-reg-Ar-joy-train.txt", fp18_reg_train_ar + "2018-EI-reg-Ar-sadness-train.txt"]

fp18_reg_dev_ar = fp18 + "reg/2018-EI-reg-Ar-dev/"
dev_reg_18_ar = [fp18_reg_dev_ar + "2018-EI-reg-Ar-anger-train.txt", fp18_reg_dev_ar + "2018-EI-reg-Ar-fear-train.txt",
                 fp18_reg_dev_ar + "2018-EI-reg-Ar-joy-train.txt", fp18_reg_dev_ar + "2018-EI-reg-En-sadness-train.txt"]

########################### SPANISH DATASETS, REG ###########################

fp18_reg_train_es = fp18 + "reg/2018-EI-reg-Es-train/"
train_reg_18_es = [fp18_reg_train_es + "2018-EI-reg-Ar-anger-train.txt", fp18_reg_train_es + "2018-EI-reg-Ar-fear-train.txt",
                 fp18_reg_train_es + "2018-EI-reg-Ar-joy-train.txt", fp18_reg_train_es + "2018-EI-reg-Ar-sadness-train.txt"]

fp18_reg_dev_es = fp18 + "reg/2018-EI-reg-Es-dev/"
dev_reg_18_es = [fp18_reg_dev_es + "2018-EI-reg-Ar-anger-train.txt", fp18_reg_dev_es + "2018-EI-reg-Ar-fear-train.txt",
                 fp18_reg_dev_es + "2018-EI-reg-Ar-joy-train.txt", fp18_reg_dev_es + "2018-EI-reg-En-sadness-train.txt"]

########################### ENGLISH DATASETS, CLASS ###########################

fp18_class_train_en = fp18 + "class/2018-EI-oc-En-train/"
train_class_18_en = [fp18_class_train_en + "2018-EI-oc-En-anger-train.txt", fp18_class_train_en + "2018-EI-oc-En-fear-train.txt",
                 fp18_class_train_en + "2018-EI-oc-En-joy-train.txt", fp18_class_train_en + "2018-EI-oc-En-sadness-train.txt"]

fp18_class_dev_en = fp18 + "class/2018-EI-oc-En-dev/"
dev_class_18_en = [fp18_class_dev_en + "2018-EI-oc-En-anger-train.txt", fp18_class_dev_en + "2018-EI-oc-En-fear-train.txt",
                 fp18_class_dev_en + "2018-EI-oc-En-joy-train.txt", fp18_class_dev_en + "2018-EI-oc-En-sadness-train.txt"]

########################### ARABIC DATASETS, CLASS ###########################

fp18_class_train_ar = fp18 + "class/2018-EI-oc-Ar-train/"
train_class_18_ar = [fp18_class_train_ar + "2018-EI-oc-Ar-anger-train.txt", fp18_class_train_ar + "2018-EI-oc-Ar-fear-train.txt",
                 fp18_class_train_ar + "2018-EI-oc-Ar-joy-train.txt", fp18_class_train_ar + "2018-EI-oc-Ar-sadness-train.txt"]

fp18_class_dev_ar = fp18 + "class/2018-EI-oc-Ar-dev/"
dev_class_18_ar = [fp18_class_dev_ar + "2018-EI-oc-Ar-anger-train.txt", fp18_class_dev_ar + "2018-EI-oc-Ar-fear-train.txt",
                 fp18_class_dev_ar + "2018-EI-oc-Ar-joy-train.txt", fp18_class_dev_ar + "2018-EI-oc-En-sadness-train.txt"]

########################### SPANISH DATASETS, CLASS ###########################

fp18_class_train_es = fp18 + "class/2018-EI-oc-Es-train/"
train_class_18_es = [fp18_class_train_es + "2018-EI-oc-Ar-anger-train.txt", fp18_class_train_es + "2018-EI-oc-Ar-fear-train.txt",
                 fp18_class_train_es + "2018-EI-oc-Ar-joy-train.txt", fp18_class_train_es + "2018-EI-oc-Ar-sadness-train.txt"]

fp18_class_dev_es = fp18 + "class/2018-EI-oc-Es-dev/"
dev_class_18_es = [fp18_class_dev_es + "2018-EI-oc-Ar-anger-train.txt", fp18_class_dev_es + "2018-EI-oc-Ar-fear-train.txt",
                 fp18_class_dev_es + "2018-EI-oc-Ar-joy-train.txt", fp18_class_dev_es + "2018-EI-oc-En-sadness-train.txt"]

reg_file_list = [train_reg_18_en, dev_reg_18_en, train_reg_18_ar, dev_reg_18_ar, train_reg_18_es, dev_reg_18_es]
class_file_list = [train_class_18_en, dev_class_18_en, train_class_18_ar, dev_class_18_ar, train_class_18_es, dev_class_18_es]


#print(class_file_list[0])

#def filepath_returner(language, year, task):
#    if task == 'reg':
#        language+year       