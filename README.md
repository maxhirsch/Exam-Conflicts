# Introducing ssmes (School of Science and Math Exam Scheduler)
This exam scheduler is the result of a NCSSM Mini-Term 2019 project intended to help the school reduce exam conflicts. The final solution of the Mini-Term was to have only two exam blocks each day so as to avoid conflicts in which students would have three exams within a 24 hour period. Thus, this scheduler reduces the number of exam conflicts where students have more than one exam in a single exam block.

## Installing ssmes
Most conveniently, SSMES can be downloaded as a single package for Windows or MacOS (files in the `Compiled Applications` folder). Alternatively, SSMES can be run in Python without a GUI through `exam_scheduler.py`; `ssmes_interface.py` and `test.py` are required for a GUI, which may be run from the command line with the command `python test.py`. 
### Windows
Simply download the `ssmes_windows` application from the `Compiled Applications` folder. The user may be prompted for administrator privileges during use (e.g. when opening help links or Github links from the help menu). Allow these privileges if prompted for full functionality.
### MacOS
Download the `ssmes_macos` application from the `Compiled Applications` folder. Before use, the user may be alerted that the application is from an untrusted developer. To fix this, go to *System Preferences &rarr; Security and Privacy &rarr; General* and trust the application at the bottom of the window.
### Dependencies
`ssmes` requires `numpy` and `pandas`. The unpackaged GUI additionally requires `PyQt5`.

## Available Settings
#### Term Schedule
Under *Settings &rarr; Term Schedule*, the school schedule can be toggled from trimester to semester.
#### Total Random Trials
Under *Settings &rarr; Total Random Trials*, the number of random initializations of schedules can be set. The default value is 5. Larger values of this setting will result in fewer conflicts, but the minimization will take a longer time.
#### Random Seed
Under *Settings &rarr; Random Seed*, the random seed for Total Random Trials can be set. The default value is 42.

## Entry Fields
#### Term
Term is the trimester (T1, T2, or T3) or semester (S1 or S2) for which you would like to create an exam schedule. The term must chosen in the list before any data can be loaded.
#### Data File
The data file field is where you input the name/location of the file which contains course enrollment data. You can browse for this data by clicking "Choose File" beside the entry field. The data file must follow a specific format which we outline below.
##### Data File Format
The data file with course enrollment data is a .csv file formatted in the following way:
<br />
<br />
StudentID,Trimester,Course1,Course2,...<br />
1,1,0,1,...<br />
2,1,0,0,...<br />
3,1,1,0,...<br />
<br />
If on a semester schedule, "Semester" would appear in the header line rather than "Trimester." Course1, Course2,... represent the course numbers with two alphabetic characters followed by three numeric characters (e.g. CH401). Each row is the student ID followed by the term (1, 2, or 3 if a Trimester schedule; 1 or 2 if Semester), and a list of 1s and 0s according to whether or not respectively a student is taking that column's course. See the file `classes_2018.csv` for an example.
#### Number of Exam Days
Number of Exam Days is the number of days on which exams will be held. If, for example, it is 4, then there will be 2*4=8 exam blocks.
#### Courses and Excluded Courses
The Courses field will be populated when you load data in the Data File field. You can then exclude courses from the minimization (e.g. classes without exams or with final papers instead of exams) by clicking the course then clicking "To Excluded." You can move an excluded course back to the Courses field by clicking on the course in the Excluded Courses field and clicking "From Excluded." Once data is loaded from the Data File field, you can also load excluded courses into the Excluded Courses field by choosing an Excluded Courses file. This can be done with the Excluded Courses File field which is found below the Excluded Courses field. This file must be formatted in a particular way which we describe below.
##### Excluded Courses File
The Excluded Courses file should be formatted such that each excluded course is on its own line with no other characters surrounding it. For example:
<br />
<br />
MA470<br />
RE120<br />
MA472<br />

and so on.
#### Departments
The Departments field will be populated when data is loaded in the Data File field. Departments can be split in the algorithm. We allow the user to explicitly split or not split departments by moving departments to the Definite Split Departments and Restricted Split Departments fields respectively in a manner similar to that of moving courses to excluded courses as described in __Courses and Excluded Courses__. The splitting of departments can also be inferred. This inference can be enabled by checking the Infer Splits checkbox below the Restricted Split Departments field.
## Output
The result of the optimization is two files. The user will provide a filename, *fn*, and a popup will allow the user to select a folder to which the files should be saved. The files will be saved as __*fn*\_department\_courses.json__ and __*fn*\_exam\_blocks.json__. *fn*_department_courses.json contains the departments and their associated courses (a split department will end with a suffix to denote this). *fn*_exam_blocks.json will contain each exam block with the departments in that exam block. Note that there is no actual ordering to the exam blocks, as the order of the blocks themselves will not affect the total number of conflicts.
