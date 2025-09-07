
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SLADS-Net is a deep neural networks version of the original work of SLADS

 Yan Zhang^, G. M. Dilshan Godaliyadda* , Nicola Ferrier#, Emine B. Gulsoy+, Charles A. Bouman*, Charudatta Phatak^
^ Materials Science Division, Argonne National Laboratory, Lemont, IL
* ECE Department, Purdue University, West Lafayette, IN
# Mathematics and Computer Science Division, Argonne National Laboratory, Lemont, IL
+ Department of Materials Science and Engineering, Northwestern University, Evanston, IL

 For SLADS-Net questions contact <yz@anl.gov> or <firegod20zy@gmail.com>


SLADS - A supervised Learning Approach to Dynamic Sampling is an algorithm 
 that allows a user to dynamically sample an image.
											
 This code and the SLADS method were developed by G.M. Dilshan P. Godaliyadda^,
 Dong Hye Ye^, Michael Uchic*, Michael Groeber*, Gregery Buzzard# and Charles 
 Bouman^  
 ^ ECE department Purdue University					
 * Mathematics department Purdue University						 
 # Material Science Directorate, Air-Force Research Laboratory Dayton, OH		 

 For questions regarding the SLADS code contact <dilshangod@gmail.com>	   		
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



###############################################################################
Instruction for Pre-trained SLADS-Net:

Below this section is the users' manual for SLADS. But in the most up-to-date SLADS-Net
version, the training phase is optional. SLADS-Net was pre-trained using cameraman
image and users can just run simulation or perform imaging experiments directly (jump to Section 2).

Reading the users' manual for original SLADS is recommended.
###############################################################################


















===============================================================================
SECTION 0: How to use this code
===============================================================================

- Install Python (version 3.5+) and the Libraries numpy, pylab, scipy, sklearn,
  matplotlib. 
  You can also install them using Anaconda package. 
  <https://www.continuum.io/downloads>
  **** If version 2.7 is used instead make sure to enter ALL INPUTS for the 
       SCRIPTS as a FLOAT instead of an int e.g. 4 should be 4.0 (applies to
       numpy arrays as well)

- Read manuscript explaining the SLADS algorithm: 
  <https://engineering.purdue.edu/~bouman/publications/orig-pdf/
  ei2016-Dilshan.pdf>

- Read manuscript explaining the SLADS-Net algorithm:
  https://arxiv.org/abs/1803.02972


- User must first train using images that look similar to the image (in 
  simulation) or object (in actual experiment) they wish to sample by following
  the instructions in Section 1

- Then to ensure the algorithm is performing properly the user is encouraged to
  run the script runSimulation.py on another image (similar to the object they
  wish to sample) by following the instructions in Section 2

- The user can now implement SLADS on an imaging device by following the 
  instructions in Section 3
===============================================================================






===============================================================================
SECTION 1: Instructions for Training (Necessary to run SLADS)
===============================================================================

This script allows the user to performing training for SLADS. 
- First the code will find training coefficients for different values of 'c' 
  and will automatically select the optimal 'c' for testing and save it for use
  in simulations or experiments. 
- The code also allows the user to find the stopping threshold for a desired
  distortion level. 

0. Select images that are:
   - similar to the intended testing data
   - the same format
   - the same size
   - not color images (i.e. only single grayscale value for each pixel)


1. Saving Images for training
	
	1.1. In folder './ResultsAndData/TrainingData/' create folder titled 
             'TrainingDB_X' (e.g 'TrainingDB_1')

	1.2. In folder './ResultsAndData/TrainingData/TrainingDB_X' create folder
         titled ‘Images’ and save images for training

	1.3. In folder './ResultsAndData/TrainingData/TrainingDB_X' create folder 
         titled 'ImagesToFindC' and  save images that will be used to chose the
         best 'c' (in the approximate RD) and the threshold on the stopping 
         function (that corresponds to a desired distortion)


2. Initializing the script to run training

	2.1 In runTraining.py go to section 'USER INPUTS: L-0' and enter 
	    information that corresponds to training data saved in 
        './ResultsAndData/TrainingData/TrainingDB_X'

            ** Double check 'ImageType' is set to 'C' for continuous and 'D' 
               for discrete images
   	    ** All entries that need to be set here are very important and need to
	       match the training images

	2.2. In runTraining.py  modify section 'USER INPUTS: L-1'
        - if mask sizes in training need to be changes
        - if the user wants to change the approximate RD summation window size 
        - if the user wants to modify ERD update window size for SLADS
        
        - if the user wants to set Total distortion value for stopping 
               condition 
        - if the initial mask type for SLADS needs to be changed        
        ** If using mask type 'H' as initial mask make sure there is a folder  
           inside'./ResultsAndData/InitialSamplingMasks' that corresponds to 
           the settings of variable 'InitialMask' with a file 
           'SampleMatrix.npy'
           e.g. if initial percentage 1% and size of image 64x64 folder name:
           H_1_64_64_Percentage_1.0
   	    ** ONLY MODIFY IF user has EXPERT knowledge of SLADS training procedure
               ELSE DO NOT CHANGE
	    ** points 1 and 2 relate to finding coefficient vector in training
	       points 3 and 4 relate to the SLADS experiments when finding c
	       point 5 relates to finding stopping threshold


3. Run runTraining.py 
(can run on the terminal by typing  >> python3 runTraining.py)

** Resulting coefficient vector will be saved in:
  './ResultsAndData/TrainingSavedFeatures/TrainingDB_X/c_n'
  where 'n' in 'c_n' is the chosen value of 'c'
===============================================================================







===============================================================================
SECTION 2: Instructions for Running a SLADS Simulation
===============================================================================

This script allows the user to run a simulation of SLADS on a desired image. 

1. Make sure training data available for simulation
	
	1.1. If Training database 'X' is to be used for experiment make sure folder
        'TrainingDB_X' exists in path ‘./ResultsAndData/TrainingSavedFeatures/'
	1.2. If the value of 'c' chosen for experiment is 'n' make sure folder 'c_n'
	     exists in path './ResultsAndData/TrainingSavedFeatures/TrainingDB_X/'

2. Save testing data

	2.1. Save one image you wish to sample in a folder named 'TestingImageSet_Y'
             (e.g. TestingImageSet_1) in path './ResultsAndData/TestingImages/'

3. Initializing the script to run simulation

	3.1 In runSLADSSimulation.py go to section 'USER INPUTS: L-0' and enter 
	    information that corresponds to:
	    - training data saved in './ResultsAndData/TrainingData/
	    TrainingDB_X'
            - testing image saved in  './ResultsAndData/TestingImages/
	    TestingImageSet_Y'
	    - simulation details (e.g. image size, stopping percentage etc.)

            ** Double check 'ImageType' is set to 'C' for continuous and 'D' 
                for discrete images
   	    ** All entries that need to be set here are very important and need to 
	       match the training and testing data
	    ** If using mask type 'H' as initial mask make sure there is a folder  
          inside'./ResultsAndData/InitialSamplingMasks' that corresponds to the
          settings of variable 'InitialMask' with a file 'SampleMatrix.npy'
          e.g. if initial percentage 1% and size of image 64x64 folder name:
          H_1_64_64_Percentage_1.0

	3.2. In runSLADSSimulation.py modify section 'USER INPUTS: L-1'
	    - if for group-wise sampling needs to be changed
	    - if the user wants to modify ERD update window size for SLADS
 
4. Run runSLADSSimulation.py 
(can run on the terminal by typing  >> python3 runSLADSSimulation.py)

** The results will be saved in:
  './ResultsAndData/SLADSSimulationResults/Folder_Name'
   where, Folder_Name is the name of the folder you entered in runSimulation.py
===============================================================================







===============================================================================
SECTION 3: Instructions for Running a SLADS on Imaging Device 
           NOTE: Will NOT run without integrating code to imaging device
===============================================================================

This script allows the user to run an actual SLADS experiment by plugging in a
measurement routine that can acquire measurements using an imaging device. 
Note this will not run until this routine is included.

1. Make sure training data available for simulation
	
	1.1. If Training database 'X' is to be used for experiment make sure folder 
        'TrainingDB_X' exists in path './ResultsAndData/TrainingSavedFeatures/'
	1.2. If the value of 'c' chosen for experiment is 'n' make sure folder 'c_n'
	     exists in path './ResultsAndData/TrainingSavedFeatures/TrainingDB_X/'

2. Initializing the script to run simulation

	2.1 In runSLADS.py go to section 'USER INPUTS: L-0' and enter information
	    that corresponds to:
	    - training data saved in './ResultsAndData/TrainingData/
	    TrainingDB_X'
	    - experiment details (e.g. image size, stopping percentage etc.)

            ** Double check 'ImageType' is set to 'C' for continuous and 'D' for 
               discrete images
   	    ** All entries that need to be set here are very important and need to 
	       match the training and testing data
	    ** If using mask type 'H' as initial mask make sure there is a folder  
          inside'./ResultsAndData/InitialSamplingMasks' that corresponds to the
          settings of variable 'InitialMask' with a file 'SampleMatrix.npy'
          e.g. if initial percentage 1% and size of image 64x64 folder name:
          H_1_64_64_Percentage_1.0
	2.2. In runSLADS.py modify section 'USER INPUTS: L-1'
	    - if for group-wise sampling needs to be changed
	    - if the user wants to modify ERD update window size for SLADS

3. Open ./code/runSLADSOnce.py and search for CODE HERE (will show two 
   locations). In each location enter the code that will perform measurements
   according the specifications.
 
4. Run runSLADS.py (can run on the terminal by typing  >> python3 runSLADS.py)

** The results will be saved in './ResultsAndData/SLADSResults/Folder_Name'.
   Here, Folder_Name is the name of the folder you entered in runSLADS.py
===============================================================================

===============================================================================
SECTION 3.5: Instructions for Running SLADS for Energy-Dispersive Spectroscopy on the Phenom ProX Scanning Electron Microscope 
===============================================================================

This script allows the user to run an actual SLADS Energy-Dispersive Spectroscopy (EDS) experiment on the Phenom ProX Scanning Electron Microscope. It is based on a measurement code that was developed in accordance with the instructions detailed in section 3.5. Note that this method only works for SLADS Energy-Dispersive Spectroscopy, where the sample is made up of some number of elemental phases. It does not work for non-EDS SLADS, since the code enabling the Phenom to acquire an SEM image pixel-by-pixel hasn't been developed yet. For questions on this section, please contact jacques.attinger@gmail.com

1. Make sure training data available for simulation
	
	1.1. If Training database 'X' is to be used for experiment make sure folder 
        'TrainingDB_X' exists in path './ResultsAndData/TrainingSavedFeatures/'
	1.2. If the value of 'c' chosen for experiment is 'n' make sure folder 'c_n'
	     exists in path './ResultsAndData/TrainingSavedFeatures/TrainingDB_X/'

2. Initializing the script to run simulation

	2.1 In runSLADS.py go to section 'USER INPUTS: L-0' and enter information
	    that corresponds to:
	    - training data saved in './ResultsAndData/TrainingData/
	    TrainingDB_X'
	    - experiment details (e.g. image size, stopping percentage etc.)

            ** Double check 'ImageType' is set to 'C' for continuous and 'D' for 
               discrete images
   	    ** All entries that need to be set here are very important and need to 
	       match the training and testing data
	    ** If using mask type 'H' as initial mask make sure there is a folder  
          inside'./ResultsAndData/InitialSamplingMasks' that corresponds to the
          settings of variable 'InitialMask' with a file 'SampleMatrix.npy'
          e.g. if initial percentage 1% and size of image 64x64 folder name:
          H_1_64_64_Percentage_1.0
	2.2. In runSLADS.py modify section 'USER INPUTS: L-1'
	    - if for group-wise sampling needs to be changed
	    - if the user wants to modify ERD update window size for SLADS

3. Set up virtual environments

   Since the version of PyPhenom (API used to communicate with the Phenom SEM) our lab uses is only compatible with Python 3.8 and below, but the SLADS training scripts utilize PyTorch (which requires Python 3.9 and above), our solution was to create two virtual environments, SLADS-Env and Phenom-Env.


	3.1 On your Phenom SEM, create two virtual environments as shown:

      ```sh
      Repositories/
      ├── SLADS-Env/
      │   └── SLADS-Net/
      └── Phenom-Env/
          └── Phenom/ (this is the folder called "Phenom" contained in this SLADS-Net Github repo)
      ```


      - In SLADS-Env, use pip to install the packages listed in the requirements.txt file directly located in SLADS-Net
      - In Phenom-Env, use pip to install the packages listed in the requirements.txt file located in the 'Phenom' subdirectory of SLADS-Net
	3.2. In runSLADS.py modify section 'USER INPUTS: L-1'
	    - if for group-wise sampling needs to be changed
	    - if the user wants to modify ERD update window size for SLADS

4. Set up FastAPI server

   Since running SLADS-EDS on the Phenom SEM requires two different virtual environments and involves running scripts from both of them concurrently, the solution we came up with was to set up a FastAPI server. Essentially, this server enables one to obtain an EDS spot measurement from any other Python script simply by passing a web URL. This allows scripts in the SLADS-Env to call scripts from the Phenom-Env. To set this up:

   4.1 Enter and activate Phenom-Env
   4.2 In your terminal, run: fastapi dev Acquire_Image.py

   - This creates a web server, which allows one to call the functions in Acquire_Image.py from anywhere simply using code like this:
      
      url = "http://127.0.0.1:8000/acquire_EDS_spot_spectrum"
    params = {
        "i": i,
        "j": j,
        "width": width,
        "height": height,
        "SavePath": full_path,
        "MainPath": MainPath,
        "sample_name": sample_name,
        "dwell_time": dwell_time
    }

    response = requests.get(url, params=params)

   - The first part of the URL signifies the FastAPI server you've created for Acquire_Image.py. The second part "acquire_EDS_spot_spectrum" signifies the function you want to call from Acquire_Image.py.  
 
5. Run runSLADS.py (can run on the terminal by typing  >> python3 runSLADS.py)

** The results will be saved in './ResultsAndData/SLADSResults/Folder_Name'.
   Here, Folder_Name is the name of the folder you entered in runSLADS.py


===============================================================================
SECTION 4: Autonomous Detection and Analysis of Nickel Microparticles (separate project frmo SLADS)
===============================================================================

Within the SLADS-Net repository, the 'Phenom' subdirectory contains several scripts related to a project distinct from SLADS that utilizes the AutoPhenom library to autonomously detect and analyze a sample of thousands of nickel microparticles under the Phenom ProX Scanning Electron Microscope. For any questions about this code, please contact jacques.attinger@gmail.com. The code for this project can be described as follows:

- RunNickelSpheresEDSAnalysis.py
   - The culmination of this project. Plug in your experiment parameters and run this file to autonomously acquire data for hundreds of nickel microparticles in one click.
   - relies on several functions in the AutoPhenom library
- PerformingEDSon500Particles.ipynb
   - Exploratory work to enable the Phenom SEM to "walk" in an orderly fashion across the sample, acquiring SEM images at each location it reaches
- ThinFilmsInvestigation.ipynb
   - Exploratory work in which we acquired images of thin films. This is the first step toward an extension of the Nickel Microparticles project that would involve defect identification/characterization for thin films.
-  NickelSphereImageSegmentation.ipynb
   - Exploratory work in which we acquired an image of several nickel particles and performed image segmentation to identify particles and acquire data such as radius and Energy-Dispersive spectra

- NickelConcentrationRadialTrend.ipynb
   - Analyzed the data acquired during our autonomous characterization of 500 nickel microparticles

4.1 To run an automated particle characterization experiment, create and utilize the Phenom-Env environment as detailed in section 3.5.3.1 of this Readme file. The script RunNickelSpheresEDSAnalysis.py can then be run from there.


===============================================================================
SECTION 5: Content
===============================================================================

./code                                : folder with code

./ResultsAndData                      : folder with data needed for experiments
                                        and where results will saved

./ResultsAndData/InitialSamplingMasks : contains in folders the initial    
                                        measurement masks of type ‘H’ i.e. 
                                        low-discrepancy sampling 
                                        e.g. if initial percentage 1% and size 
                                        of image 64x64 folder name: 
                                        H_1_64_64_Percentage_1
                                        SampleMatrix.npy: Matrix with 0's and 
                                        1's, 1 - measurement location
				
./ResultsAndData/SLADSSimulationResults: folder where SLADS simulation results 
                                        are saved

./ResultsAndData/SLADSResults          : folder where SLADS results are saved


./ResultsAndData/TestingImages         : contains images that will be used for 
                                         SLADS simulations (i.e to run 
                                         runSLADSSimulation.py)

./ResultsAndData/TrainingData          : Contains the results of training and
                                          the images used for training

./ResultsAndData/TrainingSavedFeatures : folder where coefficient vectors 
                                         computed in training will be saved	
===============================================================================
