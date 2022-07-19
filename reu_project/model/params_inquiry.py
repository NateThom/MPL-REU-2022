import inquirer as inq
from sys import stdout
from load_data import *


# get model parameters from user using inquiry module
def get_params():
    models = []
    # get number of models requested
    valid = False
    while not valid:
        try:
            num_models = int(inq.text("Number of models"))
            valid = True
        except:
            print("Invalid value. Enter an integer.")

    # for each model, get all the parameters required
    for num in range(num_models):
        # start getting parameters
        params = []
        print(f"\nParameters for model {num+1}\n=========================")

        # get model name
        params.append(inq.text("Model Name"))
        print('')

        # get new file name if needed
        new_file = inq.list_input("Save to separate model file?", choices=['no', 'yes'])
        if new_file == 'yes':
            params.append(inq.text('New model file'))
        else:
            params.append(None)

        # get True or False for train, test, load, and save
        ops = inq.checkbox("Select which operations to perform",
                           choices=['train', 'test', 'load', 'save'])
        for op in ['train', 'test', 'load', 'save']:
            params.append(True) if op in ops else params.append(False)

        # get training dataset
        if params[2]:
            dataset = inq.list_input("Training Dataset", choices=['CelebA', 'Occluded'])
            params.insert(0, {'CelebA': CelebA_Dataset, 'Occluded': Occluded_Dataset}[dataset])
        else:
            params.insert(0, CelebA_Dataset)

        # get special parameters for augmented data
        if params[0] == CelebA_Dataset:
            params.insert(1, None)
        else:
            valid = False
            while not valid:
                if params[0] == Occluded_Dataset:
                    choices = ['eyebrows', 'eyes', 'nose', 'mouth', 'chin']
                augs = inq.checkbox("Select augmentations to use",
                                    choices=choices)
                if augs == []:
                    print("Select at least one augmentation.")
                else:
                    valid = True
                    # translate selected augmentations into binary
                    augs = [1 if aug in augs else 0 for aug in
                            choices]
            portion_min = 50
            portion_max = min(100, ((162079 * ((2 ** len([i for i in augs if i == 1]))
                                               - 1)) * 100) // 324158)
            valid = False
            print('\n')
            while not valid:
                try:
                    portion = int(inq.text("Percentage of dataset to be augmented "
                                           f"({portion_min} - {portion_max})"))
                    if portion_min <= portion <= portion_max:
                        valid = True
                    else:
                        raise Exception
                except:
                    print(f"Invalid value. Enter an int from {portion_min} to {portion_max}.")
            params.insert(1, [augs, portion/100])

        # get custom dataset splits if requested
        default_split = inq.list_input("Default dataset split?", choices=['yes', 'no'])
        if default_split == 'no':
            print("Enter custom splits. Enter each as two floats between 0 and 1.")
            split_check = {'training':params[4], 'validation':params[4], 'testing':params[5]}
            for split in ['training', 'validation', 'testing']:
                if split_check[split]:
                    valid = False
                    while not valid:
                        c_split = inq.text(f"Custom {split} split")
                        try:
                            c_split = (tuple([float(v) for v in c_split.split()]))
                            if len(c_split) == 2 and 0 <= c_split[0] <= c_split[1] <= 1:
                                valid = True
                                params.append(c_split)
                            else:
                                raise Exception()
                        except:
                            print("Invalid value. Enter two floats between 0 and 1.")
                else:
                    params.append((0, 0))
            print('')
        else:
            params.extend([(0, 0.8), (0.8, 0.9), (0.9, 1)])

        # get custom learning rate and scheduler if requested
        default_lr = inq.list_input("Default learning rate and scheduler?",
                                    choices=['yes', 'no'])
        if default_lr == 'no':
            valid = False
            while not valid:
                c_ilr = inq.text("Custom learning rate")
                try:
                    params.append(float(c_ilr))
                    valid = True
                except:
                    print("Invalid value. Enter a float.")
            print('')
            c_lrs = inq.list_input('Select scheduler',
                                   choices=['step-wise', 'exponential', 'constant'])
            params.append({'step-wise':'step', 'exponential':'exp', 'constant':'const'}[c_lrs])
        else:
            params.extend([0.0001, 'step'])

        # add set of parameters to list
        models.append(params)

    # print to stdout and return
    stdout.flush()

    return models
