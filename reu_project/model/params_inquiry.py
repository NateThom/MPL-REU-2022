import inquirer as inq
from sys import stdout
from load_data import *


# get model parameters from user using inquiry module
def get_params():
    # get number of models requested
    models = []
    num_models = int(inq.text("Number of models to train"))

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

        # get training dataset
        dataset = inq.list_input("Training Dataset", choices=['CelebA', 'Occluded'])
        params.insert(0, {'CelebA': CelebA_Dataset, 'Occluded': Occluded_Dataset}[dataset])

        # get True or False for train, test, load, and save
        ops = inq.checkbox("Select which operations to perform",
                           choices=['train', 'test', 'load', 'save'])
        for op in ['train', 'test', 'load', 'save']:
            params.append(True) if op in ops else params.append(False)

        # get custom dataset splits if requested
        default_split = inq.list_input("Default dataset split?", choices=['yes', 'no'])
        if default_split == 'no':
            print("Enter custom splits. Enter each as two floats between 0 and 1")
            split_check = {'training':params[3], 'validation':params[3], 'testing':params[4]}
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
