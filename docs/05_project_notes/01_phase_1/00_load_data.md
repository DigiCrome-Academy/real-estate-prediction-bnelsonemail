# Phase 01 Project Notes
The notes in this markdown are meant to document logical thinking, problems, and solutions encountered during the 
progression of the development of this project.

## Data Loader Python File
The data_loader.py file has the following functions preloaded, but not implemented:
- load_housing_data
- preprocess_features
- split_data
- create_feature_engineering

### Download Data
The README file directed the student to download the data and provided a CLI command calling on python to run the 
data_loader.py file.  The load_housing_data function uses memory to create the dataframe, but calling on data_loader.py
implies saving a file to the data directory.  This is a direct contradiction unless one assumes the execution of the 
data_loader.py in the CLI was supposed to download and save the code.  That means there were two options:

_**Option 1**_
Create a clean function called `download_data` which downloads and saves the data in the data directory.

_**Option 2**_
Create code in the execution block at the bottom of the page that downloads the data and saves to the data directory.

_**Discussion**_
While option 2 is likely the intent of this assignment, it is not the clean way to handle this task.  Option 2 presents 
a clean and much more scalable way to handle the task.  Placing the code into a function also prevents the code from 
being executed every time the file is called.  Therefore, the decision to place the code into a function was the option
chosen.  The function added logic to check if the file already exist and also provided basic status updates to the user.

### Load Housing Data
The load housing data function directed implementation of using sklearn to fetch the data and return a dataframe
which includes all the features and the target.  Sklearn has a feature that allows this all in one line.  Setting a 
variable to calling the fetch  data using (as_frame=True) will download and store the data in memory as a `bunch object`,
which is sklearn's container for datasets.  To convert the `bunch object` to a pandas dataframe, the command `.frame` is
used to convert to a full dataframe.  There are several commands that can be used with sklearn's containers:

| Command        | Description                |
| -------------- | -------------------------- |
| data.data      | features (DataFrame)       |
| data.target    |   target (Series)          |
| data.frame     |   full dataset (DataFrame) |
| data.feature_names | feature names (string) |
| data.DESCR     | description of dataset (string) |

