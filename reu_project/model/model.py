from sys import stdout
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import time
import datetime
from math import floor
import csv
from load_data import CelebA_Dataset, Occluded_Dataset
from params_inquiry import get_params


# resnet50 class adapted from pytorch implementation
class ResNet50(nn.Module):
    def __init__(self, ilr, lrs):
        super(ResNet50, self).__init__()

        self.softmax_layer = nn.Softmax(dim=1)

        # Resnet
        self.model = torchvision.models.resnet50()

        # Edit output of fully connected layer so that only two values are output
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=2)

        # loss and optimizer
        self.loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=ilr)

        # learning rate scheduler and decay
        self.ilr = ilr
        self.lrs = lrs

    def forward(self, x):
        output = self.model(x)
        return output

    def lr(self, epoch):
        if self.lrs == 'step':
            for g in self.optimizer.param_groups:
                g['lr'] = self.ilr * (0.75 ** floor(epoch/3))
        elif self.lrs == 'exp':
            for g in self.optimizer.param_groups:
                g['lr'] = self.ilr * (0.1 ** epoch)
        else:
            pass


# get accuracy and loss during training, validation, and testing
def get_acc_and_loss(model, inputs, labels):
    # forward propagate the model
    model_output = model.forward(inputs)
    # get predictions
    prediction_probabilities = model.softmax_layer(model_output)
    predictions = prediction_probabilities.argmax(1)
    # get loss
    loss = model.loss(prediction_probabilities, labels.float())

    # get accuracy by comparing each prediction to the correct label
    correct = 0
    size = len(predictions)
    for lbl in range(size):
        pred = predictions[lbl]
        if int(labels[lbl][pred]) == 1:
            correct += 1
    acc = correct / size

    # return the accuracy and loss
    return acc, loss


# run a progress bar
def progress(prog, tickcount):
    # find the new number of progress ticks
    ticks = floor(25 * prog)
    # print the new number minus the current number
    print((ticks - tickcount) * '-', end='')
    stdout.flush()

    # if progress is complete, begin a newline
    if ticks == 25:
        print('')

    # return the new number of ticks
    return ticks


# function for each iteration of training
def train_loop(model, train_loader, val_loader, epoch, device):
    # TRAINING
    # status update
    print("Training...")
    # set model to train mode
    model.train()
    # begin records for training accuracy and loss
    trn_acc = 0
    trn_loss = 0
    # begin record for progress bar
    tickcount = 0
    # loop through batches in training data loader

    for batch_index, batch in enumerate(train_loader):
        # get inputs and labels and send them to cuda if cuda is available
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # set gradients to zero
        model.optimizer.zero_grad()

        # get accuracy and loss from get_acc_and_loss function
        acc, loss = get_acc_and_loss(model, inputs, labels)

        # backward propagate the model and update model parameters
        loss.backward()
        model.optimizer.step()

        # update records for training accuracy and loss
        trn_acc += acc
        trn_loss += loss

        # update loading bar
        tickcount = progress((batch_index + 1) / len(train_loader), tickcount)

    # find average training accuracy and loss from records
    trn_acc = "%.1f" % (trn_acc * 100 / len(train_loader))
    trn_loss = "%.3f" % (trn_loss / len(train_loader))

    # print training accuracy and loss
    print(f"Training accuracy: {trn_acc}%")
    print(f"Training loss: {trn_loss}")

    # VALIDATION
    # status update
    print("Validating...")
    # set model to evaluate mode
    model.eval()
    # begin records for validation accuracy and loss
    val_acc = 0
    val_loss = 0
    # begin record for progress bar
    tickcount = 0
    # gradients are not computed during validation
    with torch.no_grad():
        # loop through batches in validation data loader
        for batch_index, batch in enumerate(val_loader):
            # get inputs and labels and send them to cuda if cuda is available
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # get accuracy and loss from get_acc_and_loss
            acc, loss = get_acc_and_loss(model, inputs, labels)

            # update records for validation accuracy and loss
            val_acc += acc
            val_loss += loss
            
            # update loading bar
            tickcount = progress((batch_index + 1) / len(val_loader), tickcount)

        # find average validation accuracy and loss from records        
        val_acc = "%.1f" % (val_acc * 100 / len(val_loader))
        val_loss = "%.3f" % (val_loss / len(val_loader))

        # print validation accuracy and loss
        print(f"Validation accuracy: {val_acc}%")
        print(f"Validation loss: {val_loss}")

    # update learning rate
    model.lr(epoch)
    
    # return training and validation accuracy and loss
    return trn_acc, trn_loss, val_acc, val_loss


# function for testing
def test_loop(model, test_loader, device):
    # set model to evaluate mode
    model.eval()
    # begin record for testing accuracy
    acc = 0
    # begin record for progress bar
    tickcount = 0
    # gradients are not calculated during testing
    with torch.no_grad():
        # loop trough batches in test data loader
        for batch_index, batch in enumerate(test_loader):
            # get inputs and labels and send them to cuda if cuda is available
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # update record for testing accuracy
            acc += get_acc_and_loss(model, inputs, labels)[0]
            
            # update loading bar
            tickcount = progress((batch_index + 1) / len(test_loader), tickcount)
            
        # find average accuracy from records
        acc = "%.1f" % (acc * 100 / len(test_loader))
        
        # print testing accuracy
        print(f"Accuracy: {acc}%")


# big function that does all the stuff you tell it to!
def model_functions(train_set, mdl_name, new_file, train, test, load, save,
                    train_split, val_split, test_split, ilr, lrs,
                    path_to_saved_models, batch_size, device):

    # instantiate the model
    model = ResNet50(ilr, lrs)

    # if loading is specified, load the given data to the model
    if load:
        model.load_state_dict(torch.load(path_to_saved_models + mdl_name + '.mdl'))

    # send the model to cuda if cuda is available
    model = model.to(device)
    
    # if training, load the train and validation datasets and run the train loop
    if train:
        # load the training dataset. which dataset to load is a function parameter.
        train_dataset = train_set(train_split)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=10, drop_last=False, shuffle=True)

        # load the validation dataset. this is always CelebA.
        val_dataset = CelebA_Dataset(val_split)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                num_workers=10, drop_last=False, shuffle=False)

        # initialize records for training and validation loss and accuracy
        val_loss_record = []
        val_acc_record = []
        trn_loss_record = []
        trn_acc_record = []
        time_record = []
        low_vloss = 132
        # begin epochs
        epoch_index = 0
        # status report
        print(f"TRAINING {mdl_name}\n=========================\n")
        # loop training until validation loss stops decreasing
        while (epoch_index < 5 or min([float(r) for r in val_loss_record[-5:]]) <= low_vloss)\
                and epoch_index < 16:
            # increment epoch, status report and start timer
            epoch_index += 1
            print(f"Epoch {epoch_index}")
            start = time.time()

            # run training loop and capture accuracy and validation records
            records = train_loop(model, train_loader, val_loader, epoch_index, device)
            # split records among correct record lists
            val_loss_record.append(records[3])
            val_acc_record.append(records[2])
            trn_loss_record.append(records[1])
            trn_acc_record.append(records[0])

            # stop timer and print elapsed time
            finish = str(datetime.timedelta(seconds=round(time.time()-start, 2))).split(".")[0]
            time_record.append(finish)
            print("Elapsed time:", finish, '\n')

            # check for new lowest validation loss and save if required
            if float(records[3]) < low_vloss and epoch_index >= 3:
                low_vloss = float(records[3])
                if save:
                    # create an epoch-by-epoch list of loss and accuracy
                    data = [['epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss', 'time']]
                    for i in range(epoch_index):
                        data.append([i+1, trn_acc_record[i], trn_loss_record[i],
                                     val_acc_record[i], val_loss_record[i], time_record[i]])
                    # save the data list to a csv file
                    with open(path_to_saved_models+'data/'+mdl_name+'_data.csv', 'w+') as dataFile:
                        writer = csv.writer(dataFile)
                        for line in data:
                            writer.writerow(line)
                    # save the model. if a new file is specified, save to new_file
                    if new_file is None:
                        torch.save(model.state_dict(), path_to_saved_models+mdl_name+'.mdl')
                    else:
                        torch.save(model.state_dict(), path_to_saved_models+new_file+'.mdl')

    # if testing, load the test dataset and run the test loop
    if test:
        # load test dataset
        test_dataset = CelebA_Dataset(test_split)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=6, drop_last=False, shuffle=False)

        # status report
        print("TESTING")

        # run test loop
        test_loop(model, test_loader, device)

        # eventually testing data will be saved to a file.
        if save:
            pass
        

def main():
    # configuration variables
    path_to_saved_models = "../../trained_models/"

    batch_size = 64

    # select processor. if there is more than one gpu, ask which one to use
    if torch.cuda.is_available():
        device = "cuda"
        gpus = torch.cuda.device_count()
        if gpus > 1:
            while device not in [f"cuda:{n}" for n in range(gpus)]:
                device = "cuda:" + input(f"Select select gpu (0 - {gpus-1}): ")
    else:
        device = 'cpu'

    print(f"Using {device} device\n")

    # get number of models and model parameters from get_params()
    model_params = get_params()

    # run each model using given parameters
    for current_model in model_params:
        model_functions(*current_model, path_to_saved_models, batch_size, device)


if __name__ == "__main__":
    main()


# BRIEF DOCUMENTATION
#
# ------------------------------------------------------------------------------------------
# Things to change to fit your machine and stuff:
#
# line 9: Change this to import your dataset classes from your load_data.py file.
# line 307: Change the variable path_to_saved_models to wherever your model folder is.
#
# ------------------------------------------------------------------------------------------
# The main() function calls get_params(), which will prompt the user for parameter values
# for each model. If you are not using get_params(), call model_functions() manually.
# Following is a rundown of options. Defaults listed here are recommended values. They are
# selected automatically by get_params() if custom values are not specified.
#
# train_set: 
#     Input the name of the dataset class you will use for training. The class should be
#     imported in line 9. train_set defaults to CelebA_Dataset
#
# mdl_name:
#     The name of the model for saving and loading purposes.
#
# new_file:
#     If you are loading a pretrained model but saving it to a different file, set this
#     parameter to your new file name. new_file defaults to None.
#
# train:
#     Set this to True if you are training the model. train defaults to False.
#
# test:
#     Set this to True if you are testing the model. test defaults to False.
#
# load:
#     Set this to True if you are loading a pretrained model. It will load mdl_name.mdl.
#     load defaults to False.
#
# save:
#     Set this to True if you are saving your trained model and/or data. If train=True,
#     it will save the model to mdl_name.mdl, and the training data to mdl_name_data.csv.
#     save defaults to False.
#
# train_split:
#     This parameter determines what range of the dataset is split off for training. CelebA
#     has the first 80% of images allocated for training. Leaving this parameter as the default
#     (0, 0.8) will use CelebA's allocation. For custom splits, use a 2-tuple of the form
#     (split_start, split_end), where split_start and split_end are floats ranging from 0 to 1.
#
# val_split:
#     This parameter determines what range of the dataset is split off for validation. CelebA
#     has from 80% to 90% of images allocated for validation. Leaving this parameter to the
#     default (0.8, 0.9) will use CelebA's allocation. For custom splits, use a 2-tuple of the
#     form (split_start, split_end), where split_start and split_end arefloats ranging from
#     0 to 1.
#
# test_split:
#     This parameter determines what range of the dataset is split off for testing. CelebA has
#     the last 10% of images allocated for training. Leaving this parameter to the default
#     (0.9, 1) will use CelebA's allocation. For custom splits, use a 2-tuple of the form
#     (split_start, split_end), where split_start and split_end are floats ranging from 0 to 1.
#
# ilr:
#     Initial learning rate. The Adam optimizer's default learning rate is 0.001, but after
#     testing, I've set this parameter's default to 0.0001.
#
# lrs:
#     Learning rate scheduler. There are currently three implemented LRS's: exponential,
#     step-wise, and constant.
#      - Exponential divides learning rate by 10 every epoch. To select, pass lrs="exp".
#      - Step-Wise cuts learning rate by 25% every 3 epochs. To select, pass lrs="step".
#      - Constant leaves learning rate constant throughout training. To select, pass lrs="const"
#     After testing, I've set this parameter's default to "step".
#
# ------------------------------------------------------------------------------------------
# Examples of using model_functions():
#
# model_functions(Occluded_Dataset, 'occludedV1', None, True, False, False, True,
#                 (0, 0.5), (0.8, 0.84), (0.9, 0.94), 0.001, 'exp')
#
#     This will create a new model, train it on the occluded image dataset, and save it to
#     occludedV1.mdl. It will use custom training/validation/testing splits, custom initial
#     leaning rate, and exponential learning rate scheduler.
#
# model_functions(CelebA_Dataset, 'celebaV1', 'celebaV2', True, True, True, True,
#                 (0, 0.8), (0.8, 0.9), (0.9, 1), 0.0001, 'step')
#
#     This will load a model from celebaV1.mdl, train it on the celeba dataset, test it,
#     and save it to the new file celebaV2.mdl. It will use default training/validation/testing
#     splits, and the default initial learning rate and learning rate scheduler.
#
# ------------------------------------------------------------------------------------------
# Notes:
#
# Currently, all validation and testing is done using the unaugmented CelebA. This can be
# easily changed later if we need to.
#
# model_functions() will eventually have a segment which will save testing data to a file.
# currently testing data is printed to stdout, but not saved anywhere. There is a comment
# in lines 298-300, where this segment will be.
#
# Configuration variables are defined in lines 306-321. Change these as needed.
#
# Cuda will check your number of gpus and ask which one you want to use. Enter an integer in
# the range given. Indexing begins at 0.
#
# Number of epochs is no longer set, but instead is determined by validation loss during
# training. If validation loss does not decrease for five consecutive epochs, training stops.
# The validation loss check does not begin until epoch 3, due to erratic initial validation loss.
#
# When saving the model, model_functions() will only save the state at the epoch with the lowest
# validation loss. This prevents the above five-epoch training buffer from affecting the saved
# model if overfitting occurs.
#
# Model class is defined in lines 12-45. The PyTorch ResNet50 implementation is used as a base.
#
# Initial learning rate and selected learning rate scheduler are passed to the model's
# __init__() function. Learning rate is adjusted by the selected scheduler. Currently
# implemented are an exponential scheduler and a step-wise scheduler, but additional
# schedulers can be easily added. By default, get_params() passes 'step' for the
# step-wise scheduler, and an initial learning rate of 0.0001. See lines 37-45.
#
# PyCharm has a bug which prevents the use of the inquiry module in get_params(). To bypass
# this bug, set your run configuration to emulate terminal in output console. If you are
# running this program in terminal, ignore this note.
