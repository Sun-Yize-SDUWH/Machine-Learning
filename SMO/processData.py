import numpy as np


classnum = np.array([[1, 3, 11, 12, 13, 0, 0],
                     [2, 4, 6, 7, 8, 9, 14],
                     [10, 15, 0, 0, 0, 0, 0],
                     [5, 0, 0, 0, 0, 0, 0]])
classname = [['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
             ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
             ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
             ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
             ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
             ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
             ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']]
classnamenum = [8, 16, 7, 14, 6, 5, 41]
classrange = [[17, 90],
              [12285, 1484705],
              [0, 99999],
              [0, 4356],
              [1, 99]]

fOut = open("data.csv", "w")
for line in open("adult.data", "r"):
    if line.find("?") > -1:
        continue
    else:
        fStr = ''
        items = line.strip().split(", ")
        for s in range(15):
            row = np.where(classnum == s + 1)[0][0]
            col = np.where(classnum == s + 1)[1][0]
            if row == 0:
                flag = 0
                xmin = classrange[col][0]
                xmax = classrange[col][1]
                for i in range(5):
                    if xmin + (i * (xmax - xmin))/5 <= int(items[s]) <= xmin + ((i + 1) * (xmax - xmin))/5:
                        flag = i
                        break
                temp = '0,' * flag
                temp += '1,' + '0,' * (5-flag-1)
                fStr += temp
            elif row == 1:
                flag = 0
                num = classnamenum[col]
                for i in range(num):
                    if items[s] == classname[col][i]:
                        flag = i
                        break
                temp = '0,' * flag
                temp += '1,' + '0,' * (num - flag - 1)
                fStr += temp
            elif row == 2:
                if items[s] == '<=50K':
                    fStr += '-1,'
                elif items[s] == 'Male':
                    fStr += '0,'
                else:
                    fStr += '1,'
            else:
                continue
        fStr = fStr.rstrip(',')
        fStr += '\n'
        fOut.writelines(fStr)
fOut.close()
