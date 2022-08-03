import os
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def main():
    path_to_data = "/home/guest/MPL-REU-2022/TrainModel/trained_models/data/"
    models = os.listdir(path_to_data)
    models.sort()
    for i in range(len(models)):
        models[i] = models[i][:-9]
    model_dict = dict(zip(range(len(models)), models))
    for i in range(len(models)):
        print(i, model_dict[i])
    while True:
        key = input("\nselect models by numbers: ")
        try:
            if key == 'all':
                keys = [m for m in range(len(models))]
            else:
                keys = [int(k) for k in key.split()]
            try:
                selected = [model_dict[key] for key in keys]
                break
            except ValueError:
                print("please enter a number in the list")
        except ValueError:
            print("please enter integers")

    for model_name in selected:
        with open(path_to_data+model_name+'_data.csv') as d:
            reader = csv.reader(d)
            raw = list(reader)

        data = []
        for i in range(5):
            line = []
            for j in range(1, len(raw)):
                line.append(float(raw[j][i]))
            data.append(line)
        if len(raw[0]) == 6:
            timeline = [0]
            for j in range(1, len(raw)):
                val = raw[j][5].split(":")
                timeline.append((int(val[0]) * 60 + int(val[1]) + int(val[2]) / 60 + timeline[-1])//1)
            if len(data[0]) >= 10:
                inter = len(timeline) // 10
                data.append([timeline[i*inter] for i in range(10)])
            else:
                data.append(timeline)

        plt.figure(figsize=(12, 18))
        plt.suptitle(model_name)
        plt.subplots_adjust(hspace=0.5)

        plot1 = plt.subplot(2, 1, 1)
        plt.plot(data[0], data[4], '-b')
        plt.plot(data[0], data[2], '-r')
        plt.title("Training and Validation Loss")
        tl_patch = patches.Patch(color='red', label='training loss')
        vl_patch = patches.Patch(color='blue', label='validation loss')
        plt.legend(handles=[tl_patch, vl_patch])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plot1.set_ylim(ymin=0, ymax=0.5)
        plot1.set_xlim(xmin=1, xmax=len(data[0]))
        if len(data[0]) >= 10:
            plot1.xaxis.set_major_locator(MultipleLocator((len(data[0]))//10))
        plot1.xaxis.set_major_formatter(FormatStrFormatter('% 1.0f'))
        if len(raw[0]) == 6:
            taxis1 = plot1.twiny()
            taxis1.set_xlabel('time (minutes)')
            taxis1.set_xticks((data[5]))

        plot2 = plt.subplot(2, 1, 2)
        plt.plot(data[0], data[3], '-b')
        plt.plot(data[0], data[1], '-r')
        plt.title("Training and Validation Accuracy")
        ta_patch = patches.Patch(color='red', label='training accuracy')
        va_patch = patches.Patch(color='blue', label='validation accuracy')
        plt.legend(handles=[ta_patch, va_patch])
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plot2.set_xlim(xmin=1, xmax=len(data[0]))
        if len(data[0]) >= 10:
            plot2.xaxis.set_major_locator(MultipleLocator((len(data[0]))//10))
        plot2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plot2.set_ylim(ymin=0, ymax=100)
        plot2.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
        if len(raw[0]) == 6:
            taxis2 = plot2.twiny()
            taxis2.set_xlabel('time (minutes)')
            taxis2.set_xticks((data[5]))
        plt.show()


if __name__ == '__main__':
    main()
