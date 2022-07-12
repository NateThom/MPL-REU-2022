import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import time
import datetime
from math import floor
import csv
from load_data import CelebADataset, TrainDataset, TestDataset, ValidateDataset


# resnet50 class adapted from pytorch implementation
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.softmax_layer = nn.Softmax(dim=1)

        # Resnet
        self.model = torchvision.models.resnet50()

        # Edit output of fully connected layer so that only two values are output
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=2)

        # loss and optimizer. learning rate has been reduced.
        self.loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)

    def forward(self, x):
        output = self.model(x)
        return output


# get accuracy and loss during training, validation, and testing
def get_acc_and_loss(model, inputs, labels):
    # forward propogate the model
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

    # if progress is complete, begin a newline
    if ticks == 25:
        print('')

    # return the new number of ticks
    return ticks


# function for each iteration of training
def train_loop(model, train_loader, val_loader):
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

    # return training and validation accuracy and loss
    return trn_acc, trn_loss, val_acc, val_loss


# function for testing
def test_loop(model, test_loader):
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
def model_functions(train_set=CelebADataset, model_file="xyzzy", new_file=None,
                    train=False, test=False, load=False, save=False,
                    train_split=(0, 0.8), val_split=(0.8, 0.9), test_split=(0.9, 1)):
    # instantiate the model
    model = ResNet50()

    # if loading is specified, load the given data to the model
    if load:
        model.load_state_dict(torch.load(path_to_saved_models + model_file + '.mdl'))

    # send the model to cuda if cuda is available
    model = model.to(device)

    # if training, load the train and validation datasets and run the train loop
    if train:
        # load the training dataset. which dataset to load is a function parameter.
        train_dataset = train_set(train_split)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=10, drop_last=False, shuffle=True)

        # load the validation dataset. this is always CelebA.
        val_dataset = CelebADataset(val_split)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                num_workers=10, drop_last=False, shuffle=False)

        # initialize records for training and validation loss and accuracy
        val_loss_record = []
        val_acc_record = []
        trn_loss_record = []
        trn_acc_record = []
        # begin epochs
        epoch_index = 0
        # status report
        print("TRAINING\n=========================\n")
        # loop training until validation loss stops decreasing
        while len(set(val_loss_record[-3:])) != 1 or epoch_index <= 5:
            # increment epoch, status report and start timer
            epoch_index += 1
            print(f"Epoch {epoch_index}")
            start = time.time()

            # run training loop and capture accuracy and validation records
            records = train_loop(model, train_loader, val_loader)
            # split records among correct record lists
            val_loss_record.append(records[3])
            val_acc_record.append(records[2])
            trn_loss_record.append(records[1])
            trn_acc_record.append(records[0])

            # stop timer and print elapsed time
            finish = str(datetime.timedelta(seconds=round(time.time() - start, 2))).split(".")[0]
            print("Elapsed time:", finish, '\n')

    # if testing, load the test dataset and run the test loop
    if test:
        # load test dataset
        test_dataset = CelebADataset(test_split)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=6, drop_last=False, shuffle=False)

        # status report
        print("TESTING")

        # run test loop
        test_loop(model, test_loader)

    # if save is specified, save training data, testing data, or both
    if save:
        # if the model was trained, save the trained model and the training data
        if train:
            # save the model. if a new file is specified, save to new_file
            if new_file == None:
                torch.save(model.state_dict(), path_to_saved_models + model_file + '.mdl')
            else:
                torch.save(model.state_dict(), path_to_saved_models + new_file + '.mdl')

            # create an epoch-by-epoch list of loss and accuracy
            data = [['epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss']]
            for i in range(epoch_index):
                data.append([i + 1, trn_acc_record[i], trn_loss_record[i],
                             val_acc_record[i], val_loss_record[i]])
            # save the data list to a csv file
            with open(path_to_saved_models + 'data/' + model_file + '_data.csv', 'w+') as dataFile:
                writer = csv.writer(dataFile)
                for line in data:
                    writer.writerow(line)

        # There will be another conditional for saving testing data here.
        # if test:
        #     save the results of testing


# configuration variables
path_to_saved_models = "../../trained_models/"

random_seed = 128
batch_size = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")

labels_map = {0: 'female', 1: 'male'}

# Make models! Place your model_functions() calls here to perform model-related tasks.


### BRIEF DOCUMENTATION ###
#
# ------------------------------------------------------------------------------------------
# Things to change to fit your machine and stuff:
#
# line 9: Change this to import your dataset classes from your load_data.py file.
# line 281: Change the variable path_to_saved_models to wherever your model folder is.
#
# ------------------------------------------------------------------------------------------
# To do anything with models, call model_functions(). Following is a rundown of options.
#
# train_set:
#     Optional parameter. Input the name of the dataset class you will use for training.
#     The class should be imported in line 9. train_set defaults to CelebA_Dataset
#
# model_file:
#     Optional parameter. The name of the model for saving and loading purposes. model_file
#     defaults to dummy value "xyzzy" in case you forget to specify a name.
#
# new_file:
#     Optional parameter. If you are loading a pretrained model but saving it to a different
#     file, set this parameter to your new file name. new_file defaults to None.
#
# train:
#     Optional parameter. Set this to True if you are training the model. train defaults
#     to False
#
# test:
#     Optional parameter. Set this to True if you are testing the model. test defaults
#     to False
#
# load:
#     Optional parameter. Set this to True if you are loading a pretrained model. It will
#     load model_file.mdl. load defaults to False.
#
# save:
#     Optional parameter. Set this to True if you are saving your trained model and/or data.
#     If train=True, it will save the model to model_file.mdl, and the training data to
#     model_file_data.csv. save defaults to False.
#
# train_split:
#     Optional paramter. This parameter determines what range of the dataset is split off for
#     training. CelebA has the first 80% of images allocated for training. Leaving this
#     parameter to the default (0, 0.8) will use CelebA's allocation. For custom splits,
#     use a 2-tuple of the form (split_start, split_end), where split_start and split_end are
#     floats ranging from 0 to 1.
#
# val_split:
#     Optional paramter. This parameter determines what range of the dataset is split off for
#     validation. CelebA has from 80% to 90% of images allocated for validation. Leaving this
#     parameter to the default (0.8, 0.9) will use CelebA's allocation. For custom splits,
#     use a 2-tuple of the form (split_start, split_end), where split_start and split_end are
#     floats ranging from 0 to 1.
#
# test_split:
#     Optional paramter. This parameter determines what range of the dataset is split off for
#     testing. CelebA has the last 10% of images allocated for training. Leaving this
#     parameter to the default (0.9, 1) will use CelebA's allocation. For custom splits,
#     use a 2-tuple of the form (split_start, split_end), where split_start and split_end are
#     floats ranging from 0 to 1.
#
# ------------------------------------------------------------------------------------------
# Examples of using model_functions():
#
# model_functions(train_set=Occluded_dataset, model_file='occludedV1', train=True, save=True)
#     This will create a new model, train it on the occluded image dataset, and save it to
#     occludedV1.mdl. it will use default training/validation/testing splits from CelebA.
#
# model_functions(model_file='celebaV1', test=True, load=True, test_split=(0.5, 1))
#     This will load an already trained model from celebaV1.mdl and test it. It will use
#     a custom split so that the entire last half of CelebA is used for testing.
#
# ------------------------------------------------------------------------------------------
# Notes:
#
# Currently, all validation and testing is done using the unaugmented CelebA. This can be
# easily changed later if we need to.
#
# model_functions() will eventually have a segment which will save testing data to a file.
# currently testing data is printed to stdout, but not saved anywhere. There is a comment
# in lines 275-277, where this segment will be.
#
# Configuration variables are defined in lines 280-289. Change these as needed.
#
# Number of epochs is no longer set, but instead is determined by validation loss during
# training. If validation loss does not change for three consecutive epochs, training stops.
#
# Model class is defined in lines 12-30. The PyTorch ResNet50 implementation is used. Adam
# optimizer's learning rate has been reduced by a factor of 100.
#
# Learning rate will be adjusted later.