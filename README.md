## Usage 

The python file for swapping features is `swap_features.py`, in combination with a `run_swap.py` file to allow you to easily input the current feature swapping generation. Each run of the `run_swap.py` file will create a dataset for a single attribute/combination of attributes copied from source images, pasted onto destination images.

When you open it up, there should be three sections that you must edit for your system:
-On line 339 add your source landmarks file: `file_source = open('FILENAME.csv')`
-On line 343 add your destination landmarks file: `file_dest = open('FILENAME.csv')`

When you run the file, 
