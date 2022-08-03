import pandas as pd
import os
import inquirer as inq


def get_acc_of_set():
    # get test data from directory
    path = '../../trained_models/test_records/'
    files = [i for i in os.listdir(path) if i[:4] == 'occl' and 'lfw' not in i]
    files.sort()

    # get correct models from user
    augs = inq.checkbox("Select Occlusions", choices=['eyebrows', 'eyes', 'nose', 'mouth', 'chin'])
    augs = [['eyebrows', 'eyes', 'nose', 'mouth', 'chin'].index(aug) for aug in augs]
    nums = inq.checkbox("Select numbers of augmentations",
                        choices=[str(i+1) for i in range(5) if i >= len(augs)-1])
    if len(nums) > 0:
        nums = [int(i) for i in nums]
    else:
        nums = [1, 2, 3, 4, 5]
    # if nums != [1]:
    #     perc = inq.checkbox("Select augmentation percentages", choices=['50', '75', '100'])
    #     if not perc:
    #         perc = ['50', '75', '100']
    # else:
    # perc = ['50']

    # get only selected models
    selected = []
    for model in files:
        needed = True
        parts = model.split("_")
        for aug in augs:
            if parts[1][aug] != '1':
                needed = False
        if parts[1].count('1') not in nums:
            needed = False
        # if parts[2] not in perc:
        #     needed = False
        if needed:
            selected.append(model)

    print(selected)

    # get average accuracy of selected models
    acc_list = []
    for model in range(len(selected)):
        test_df = pd.read_csv(path + selected[model])

        correct = test_df.value_counts(test_df['accuracy'])
        acc_list.append(correct[1] / len(test_df))

    for m in range(len(selected)):
        print(selected[m], acc_list[m])
    set_acc = sum(acc_list) / len(acc_list)

    return round(set_acc, 3)


for i in range(35):
    print(get_acc_of_set(), '\n')
