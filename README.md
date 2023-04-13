Download the submission folder
Python version: 3.8.5
Running on conda 4.9.2

Open the Terminal and get in the submission folder by typing the following
	$ cd ../submission
Please notice that you need to type the folder where you download the submission folder (for example cd User/Desktop/submission)
Type
	$ pip install -r requirements.txt
in order to install all the needed external modules

Please also note that you need to have a Figures folder, where the figures will be solved

Open MAIN.py (Spyder) or run in the terminal by typing
 $ python3 MAIN.py

Figures of the results will show up and will be automatically saved in the folder submission/Figures

Location Search: Run the script to obtain Fig 4E,4F and 4G.
Please notice that these scripts were run on a subset of the whole dataset (only 2 mappings in the follow-up), reason for which they are not equal to the main figure.
sbjTime = this is a number indicating the day after which the mappings are considered to belong to a follow-up. 
sbjTimeRemove = this is a number indicating the threshold day before which we want to remove mappings. For instance mappings made right after the implantation can be very unstable (due to inflammatory reactions). 

Please notice that in the case of the threshold search, we are only using 2 AS and tracking some of the mapping days to find the perceptual threshold.
For visualization purposes, only N = 5 repetitions of the algorithm are done (to speed up the process), but this can easily be changed in the function gpbo_threshold(). Please notice that due to the limited size of the dataset, results are not representative. 

