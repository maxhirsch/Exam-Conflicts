import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from ssmes_interface import Ui_MainWindow
import pandas as pd
import exam_scheduler as es
import numpy as np
import webbrowser
import json
import os


def browse_data_file():
    filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', './')
    if filename:
        ui.DataFileTextEdit.setPlainText(filename)
        ui.data_filename = filename

def browse_excluded_courses_file():
    if ui.CoursesListWidget.count() + ui.ExcludedCoursesListWidget.count() == 0:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Load data first")
        msg.setInformativeText("Data must be loaded before courses can be excluded.")
        msg.setWindowTitle("Load data")
        msg.setStandardButtons(QMessageBox.Ok)
        choice = msg.exec_()
        return

    filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', './')
    if filename:
        ui.ExcludedCoursesFileTextEdit.setPlainText(filename)
        ui.excluded_courses_filename = filename



def populate_term_list_widget():
    ui.TermListWidget.addItem("T1")
    ui.TermListWidget.addItem("T2")
    ui.TermListWidget.addItem("T3")
    ui.TermSchedule = "trimester"


def move_to_excluded_courses(item=None):
    if type(item) is not QListWidgetItem:
        item = ui.CoursesListWidget.takeItem(ui.CoursesListWidget.currentRow())
        if item is None:
            return
    i = 0
    while i < ui.ExcludedCoursesListWidget.count():
        current = ui.ExcludedCoursesListWidget.item(i)
        if item <= current:
            ui.ExcludedCoursesListWidget.insertItem(i, item)
            return        
        i += 1
    ui.ExcludedCoursesListWidget.insertItem(i, item)


def move_to_included_courses():
    item = ui.ExcludedCoursesListWidget.takeItem(ui.ExcludedCoursesListWidget.currentRow())
    if item is None:
        return
    i = 0
    while i < ui.CoursesListWidget.count():
        current = ui.CoursesListWidget.item(i)
        if item <= current:
            ui.CoursesListWidget.insertItem(i, item)
            return
        i += 1
    ui.CoursesListWidget.insertItem(i, item)

def move_to_definite_splits():
    item = ui.DepartmentsListWidget.takeItem(ui.DepartmentsListWidget.currentRow())
    if item is None:
        return
    i = 0
    while i < ui.DefiniteSplitDepartmentsListWidget.count():
        current = ui.DefiniteSplitDepartmentsListWidget.item(i)
        if item <= current:
            ui.DefiniteSplitDepartmentsListWidget.insertItem(i, item)
            return
        i += 1
    ui.DefiniteSplitDepartmentsListWidget.insertItem(i, item)

def move_from_definite_splits():
    item = ui.DefiniteSplitDepartmentsListWidget.takeItem(ui.DefiniteSplitDepartmentsListWidget.currentRow())
    if item is None:
        return
    i = 0
    while i < ui.DepartmentsListWidget.count():
        current = ui.DepartmentsListWidget.item(i)
        if item <= current:
            ui.DepartmentsListWidget.insertItem(i, item)
            return
        i += 1
    ui.DepartmentsListWidget.insertItem(i, item)

def move_to_restricted_splits():
    item = ui.DepartmentsListWidget.takeItem(ui.DepartmentsListWidget.currentRow())
    if item is None:
        return
    i = 0
    while i < ui.RestrictedSplitDepartmentsListWidget.count():
        current = ui.RestrictedSplitDepartmentsListWidget.item(i)
        if item <= current:
            ui.RestrictedSplitDepartmentsListWidget.insertItem(i, item)
            return
        i += 1
    ui.RestrictedSplitDepartmentsListWidget.insertItem(i, item)

def move_from_restricted_splits():
    item = ui.RestrictedSplitDepartmentsListWidget.takeItem(ui.RestrictedSplitDepartmentsListWidget.currentRow())
    if item is None:
        return
    i = 0
    while i < ui.DepartmentsListWidget.count():
        current = ui.DepartmentsListWidget.item(i)
        if item <= current:
            ui.DepartmentsListWidget.insertItem(i, item)
            return
        i += 1
    ui.DepartmentsListWidget.insertItem(i, item)

def write_to_message_center(message):
    ui.MessageCenterTextEdit.appendPlainText(message)

def quit_application():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText("Are you sure you want to quit?")
    msg.setInformativeText("All settings will be lost.")
    msg.setWindowTitle("Quit SSMES")
    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    choice = msg.exec_()
    if choice == QMessageBox.Ok:
        QApplication.quit()

def set_semester_or_trimester():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.NoIcon)
    msg.setText("Set Term")
    msg.setInformativeText("Set term to either semester or trimester")
    msg.setWindowTitle("Set Term")
    semester_button = msg.addButton("Semester", QMessageBox.YesRole)
    trimester_button = msg.addButton("Trimester", QMessageBox.YesRole)

    choice = msg.exec_()
    ui.TermListWidget.clear()

    if choice == 0: # semester button
        ui.TermListWidget.addItem("S1")
        ui.TermListWidget.addItem("S2")
        ui.TermSchedule = "semester"
    else:
        ui.TermListWidget.addItem("T1")
        ui.TermListWidget.addItem("T2")
        ui.TermListWidget.addItem("T3")
        ui.TermSchedule = "trimester"

def set_number_random_trials():
    num, ok = QInputDialog.getInt(None, "Random Trials", "Enter number of random trials:")
    if ok:
        ui.number_random_trials = num
        write_to_message_center("Number of random trials set to {}".format(ui.number_random_trials))

def set_random_seed():
    seed, ok = QInputDialog.getInt(None, "Random Seed", "Enter a random seed:")
    if ok:
        ui.random_seed = seed
        write_to_message_center("Random seed set to {}".format(ui.random_seed))

def open_project_github():
    webbrowser.open("https://github.com/maxhirsch/ssmes")

def about_popup():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText("About SSMES")
    msg.setInformativeText("The algorithm behind SSMES was created during Mini-Term at NCSSM in 2019 to help reduce exam conflicts. Students involved in the formulation of the algorithm were Daniel Carter, Edward Feng, Kathleen Hablutzel, and Max Hirsch. All code for the project is on Github (linked in the help menu) and is free to use under a MIT License.")
    msg.setWindowTitle("About SSMES")
    msg.setStandardButtons(QMessageBox.Close)
    choice = msg.exec_()

def load_data():
    if ui.term is None:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Select term first")
        msg.setInformativeText("A term must be selected before data can be loaded")
        msg.setWindowTitle("Select term")
        msg.setStandardButtons(QMessageBox.Ok)
        choice = msg.exec_()
        return
    
    data = pd.read_csv(ui.DataFileTextEdit.toPlainText())
    if ui.TermSchedule == 'trimester':
        if "Trimester" not in list(data):
            write_to_message_center("ERROR: 'Trimester' not a column in data.")
            return
        data = data[data.Trimester == ui.term]
        data = data.drop(columns=["Student ID", "Trimester"])
    else:
        if "Semester" not in list(data):
            write_to_message_center("ERROR: 'Semester' not a column in data.")
            return
        data = data[data.Semester == ui.term]
        data = data.drop(columns=["Student ID", "Semester"])
    write_to_message_center("Loaded data from " + ui.DataFileTextEdit.toPlainText())
    ui.CoursesListWidget.clear()
    ui.ExcludedCoursesListWidget.clear()
    ui.DepartmentsListWidget.clear()
    ui.DefiniteSplitDepartmentsListWidget.clear()
    ui.RestrictedSplitDepartmentsListWidget.clear()
    courses = list(data)
    
    zero_exams = []
    for course in courses:
        # this sum is the number of people taking this course
        if sum(data[course]) == 0:
            zero_exams.append(course)
    
    # the actual removing of classes which nobody is taking
    data = data.drop(columns=zero_exams)
    for course in zero_exams:
        courses.remove(course)

    # add courses to course list widget
    for course in courses:
        i = 0
        while i < ui.CoursesListWidget.count():
            current = ui.CoursesListWidget.item(i)
            if course <= current.text():
                break
            i += 1
        ui.CoursesListWidget.insertItem(i, course)
    
    # add departments to department list widget
    departments = es.get_departments(courses)
    ui.departments = departments
    for department in list(departments):
        i = 0
        while i < ui.DepartmentsListWidget.count():
            current = ui.DepartmentsListWidget.item(i)
            if department <= current.text():
                break
            i += 1
        ui.DepartmentsListWidget.insertItem(i, department)


    ui.data = data
    ui.courses = courses

def load_excluded_courses_data():
    if ui.excluded_courses_filename is not None:
        with open(ui.excluded_courses_filename, 'r') as infile:
            excluded_courses = []
            for line in infile:
                excluded_courses.append(line.strip())
            excluded_courses.sort()
            courses = [str(ui.CoursesListWidget.item(i).text()) for i in range(ui.CoursesListWidget.count())]
            current_excluded_courses = [str(ui.ExcludedCoursesListWidget.item(i).text()) for i in range(ui.ExcludedCoursesListWidget.count())]
            for course in excluded_courses:
                if course in courses and course not in current_excluded_courses:
                    move_to_excluded_courses(ui.CoursesListWidget.takeItem(courses.index(course)))
                    courses.remove(course)

def get_number_exam_days():
    try:
        days = int(ui.NumberOfExamDaysTextEdit.toPlainText())
        return days
    except:
        write_to_message_center("ERROR: Invalid input for number of exam days.")
        return None

def get_infer_splits():
    return ui.CheckBox.isChecked()

def set_term(item):
    ui.CoursesListWidget.clear()
    ui.ExcludedCoursesListWidget.clear()
    ui.DepartmentsListWidget.clear()
    ui.DefiniteSplitDepartmentsListWidget.clear()
    ui.RestrictedSplitDepartmentsListWidget.clear()
    ui.courses = None
    ui.data = None
    ui.departments = None

    if ui.term is not None:
        current_selection = ui.term_item
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Are you sure you want to change the term?")
        msg.setInformativeText("Changing the term will reset other fields.")
        msg.setWindowTitle("Change term")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        choice = msg.exec_()
        if choice == QMessageBox.Ok:
            ui.term = item.text()
            ui.term_item = item
            write_to_message_center("Term changed and fields reset")
        else:
            ui.TermListWidget.setCurrentItem(current_selection)
    else:
        ui.term = item.text()
        ui.term_item = item

def get_term():
    return ui.term

def get_split_constraints():
    definite_splits = [str(ui.DefiniteSplitDepartmentsListWidget.item(i).text()) for i in range(ui.DefiniteSplitDepartmentsListWidget.count())]
    restricted_splits = [str(ui.RestrictedSplitDepartmentsListWidget.item(i).text()) for i in range(ui.RestrictedSplitDepartmentsListWidget.count())]
    return definite_splits, restricted_splits

def get_exam_schedule():
    """
    filename (str): The name of the data file
    """
    
    number_random_trials = ui.number_random_trials
    random_seed = ui.random_seed
    number_exam_days = get_number_exam_days()
    infer_splits = get_infer_splits()
    definite_splits, restricted_splits = get_split_constraints()

    if type(number_random_trials) is not int:
        return
    if type(random_seed) is not int:
        return
    if type(number_exam_days) is not int:
        return

    filename, ok = QInputDialog.getText(None, "Choose filename", "Enter a name for the schedule files:")
    directory = str(QtWidgets.QFileDialog.getExistingDirectory(None, 'Choose folder', './'))
    
    ############################## ACTUAL COMPUTATION ########################################


    # seed any randomness for reproducibility
    np.random.seed(random_seed)

    # get data
    data = ui.data
    courses = ui.courses

    # get departments
    departments = ui.departments

    # split departments ("splitted" b/c naming collisions...)
    splitted_departments = es.split_departments(data, departments, courses, definite_splits=definite_splits, 
            restricted_splits=restricted_splits, infer_splits=infer_splits)

    # assign departments to exam blocks
    exam_blocks = es.assign_exam_blocks(data, departments, splitted_departments, number_exam_days=number_exam_days)

    total_conflicts = sum([es.count_conflicts(data, departments, block) for block in exam_blocks])
    progress_counter = 100/number_random_trials
    ui.ProgressBar.setValue(progress_counter)
    write_to_message_center("Number of Potential Conflicts: {}".format(total_conflicts))

    # reassign departments to exam blocks and keep assignment with fewest conflicts
    for trial in range(number_random_trials-1):
        current_exam_blocks = es.assign_exam_blocks(data, departments, splitted_departments, number_exam_days=4)
        current_conflicts = sum([es.count_conflicts(data, departments, block) for block in current_exam_blocks])
        if current_conflicts < total_conflicts:
            exam_blocks = current_exam_blocks
            total_conflicts = current_conflicts
        progress_counter += 100/number_random_trials
        ui.ProgressBar.setValue(progress_counter)
        write_to_message_center("Number of Potential Conflicts: {}".format(total_conflicts))

    while os.path.exists(os.path.join(directory, "{}_department_courses.json".format(filename))) or \
          os.path.exists(os.path.join(directory, "{}_exam_blocks.json".format(filename))):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Overwrite File Warning")
        msg.setInformativeText("Are you sure you want to overwrite {} and/or {}?".format(os.path.join(directory, filename + "_department_courses.json"), os.path.join(directory, filename + "_exam_blocks.json")))
        msg.setWindowTitle("Overwrite File Warning")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        choice = msg.exec_()
        if choice == QMessageBox.No:
            filename, ok = QInputDialog.getText(None, "Choose filename", "Enter a name for the schedule files:")
        else:
            break

    with open(os.path.join(directory, "{}_department_courses.json".format(filename)), "w") as outfile:
        json.dump(departments, outfile, indent='\t')
    with open(os.path.join(directory, "{}_exam_blocks.json".format(filename)), "w") as outfile:
        blocks = {}
        for i, block in enumerate(exam_blocks):
            blocks[i+1] = block
        json.dump(blocks, outfile, indent='\t')

    print(exam_blocks)
    print(departments)#splitted_departments)
    write_to_message_center("Schedule saved to file.")

def get_help():
    if os.path.exists("./help-file/help.pdf"):
        write_to_message_center("Opening help file.")
        webbrowser.open("file://{}".format(os.path.join(os.path.dirname(os.path.abspath(__file__)), './help-file/help.pdf')))
    else:
        write_to_message_center("Help file not found. Redirecting to project Github help file.")
        webbrowser.open("https://github.com/maxhirsch/ssmes/blob/master/help-file/help.pdf")

if __name__=='__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)

    # allow file browsing
    ui.DataFileButton.clicked.connect(browse_data_file)
    ui.ExcludedCoursesFileButton.clicked.connect(browse_excluded_courses_file)

    # populate term choices with T1, T2, T3 or S1, S2
    populate_term_list_widget()

    # set quit action in file menu
    ui.actionQuit.triggered.connect(quit_application)
    ui.actionSSMES_Help.triggered.connect(get_help)
    ui.actionSSMES_on_Github.triggered.connect(open_project_github)
    ui.actionAbout_SSMES.triggered.connect(about_popup)
    ui.actionRandom_Trials.triggered.connect(set_number_random_trials)
    ui.actionRandom_Seed.triggered.connect(set_random_seed)
    ui.actionTerm_Schedule.triggered.connect(set_semester_or_trimester)

    # move items between included and excluded courses
    ui.CoursesListWidget.itemDoubleClicked.connect(move_to_excluded_courses)
    ui.ExcludedCoursesListWidget.itemDoubleClicked.connect(move_to_included_courses)

    # move items between included and excluded courses with buttons
    ui.ToExcludedButton.clicked.connect(move_to_excluded_courses)
    ui.FromExcludedButton.clicked.connect(move_to_included_courses)

    # move items between departments and definite split departments with buttons
    ui.ToDefiniteButton.clicked.connect(move_to_definite_splits)
    ui.FromDefiniteButton.clicked.connect(move_from_definite_splits)

    # move items between departments and restricted split departments with buttons
    ui.ToRestrictedButton.clicked.connect(move_to_restricted_splits)
    ui.FromRestrictedButton.clicked.connect(move_from_restricted_splits)

    ui.TermListWidget.itemActivated.connect(set_term)

    # variables for exam schedule creation

    data = None # pandas DataFrame for data file
    term = None
    number_of_exam_days = -1
    infer_split_departments = True
    departments = {}
    definite_splits = []
    restricted_splits = None

    # load data
    ui.LoadDataButton.clicked.connect(load_data)
    ui.ExcludedCoursesLoadFileButton.clicked.connect(load_excluded_courses_data)


    ui.CreateExamScheduleButton.clicked.connect(get_exam_schedule)

    # run app
    window.show()
    sys.exit(app.exec_())

