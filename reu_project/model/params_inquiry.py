import inquirer as inq
from sys import stdout
from load_data import *
import os


# get model parameters from user using inquiry module
def get_params():
    models = []
    # get number of models requested
    valid = False
    while not valid:
        try:
            num_models = inq.text("Number of models")
            if num_models != 'all':
                num_models = int(num_models)
                all = False
            else:
                num_models = 1
                all = True
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
            dataset = inq.list_input("Training Dataset",
                                     choices=['CelebA', 'Occluded', 'Standard Augmentations'])
            params.insert(0, {'CelebA': CelebADataset,
                              'Occluded': OccludedDataset,
                              'Standard Augmentations': StandardAugDataset}[dataset])
        else:
            params.insert(0, CelebADataset)

        # get testing dataset
        if params[4]:
            dataset = inq.list_input("Testing Dataset", choices=['CelebA', 'HEAT'])
            params.insert(1, {'CelebA': CelebADataset, 'HEAD': HEADDataset}[dataset])
        else:
            params.insert(1, CelebADataset)

        # get special parameters for augmented data
        if params[0] == CelebADataset:
            params.insert(2, None)
        else:
            if params[0] == OccludedDataset:
                choices = ['eyebrows', 'eyes', 'nose', 'mouth', 'chin']
            elif params[0] == StandardAugDataset:
                choices = ['rotation', 'erasing', 'blur', 'jitter', 'resize/crop']
            augs = inq.checkbox("Select augmentations to use",
                                choices=choices)
            if augs == []:
                augs = [0, 0, 0, 0, 0]
            else:
                # translate selected augmentations into binary
                augs = [1 if aug in augs else 0 for aug in
                        choices]
            params.insert(2, augs)

        # get custom dataset splits if requested
        default_split = inq.list_input("Default dataset split?", choices=['yes', 'no'])
        if default_split == 'no':
            print("Enter custom splits. Enter each as two floats between 0 and 1.")
            split_check = {'training':params[5], 'validation':params[5], 'testing':params[6]}
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
        if params[5]:
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
        else:
            params.extend([1, 'xyzzy'])

        # add set of parameters to list
        models.append(params)

        # if all models are selected, do 'em all
        if all and models[0][6]:
            allmodels = os.listdir('../../trained_models/models')
            allmodels.sort()
            for m in allmodels:
                models.append(models[0].copy())
                models[-1][3] = m[:-4]
            models.remove(models[0])

    # print to stdout and return
    stdout.flush()

    return models
