import pandas as pd
import numpy as np

def main():
    # seed any randomness
    np.random.seed(42)

    # get data
    data, courses = load_data("classes_2018.csv", "T2")

    # get departments
    departments = get_departments(courses)
    
    # split departments
    splitted_departments = split_departments(data, departments, courses, definite_splits=["PH", "CH", "BI", "MA"])

    # assign departments to exam blocks
    exam_blocks = assign_exam_blocks(data, departments, splitted_departments, number_days=4)
    total_conflicts = sum([count_conflicts(data, departments, block) for block in exam_blocks])

    for trial in range(5):
        current_exam_blocks = assign_exam_blocks(data, departments, splitted_departments, number_days=4)
        current_conflicts = sum([count_conflicts(data, departments, block) for block in current_exam_blocks])
        if current_conflicts < total_conflicts:
            exam_blocks = current_exam_blocks
            total_conflicts = current_conflicts

    print(exam_blocks)
    print(total_conflicts)

    return None

def load_data(filename, trimester):
    """
    filename: the name of the data file as a string
    trimester: the name of the trimester as a string ("T1", "T2", "T3")

    returns: a tuple of a pandas dataframe and list of courses
    """

    # load data
    data = pd.read_csv(filename)

    # filter data to desired trimester
    data = data[data.Trimester == trimester]
    data = data.drop(columns=["Student ID", "Trimester"])

    # get list of courses
    courses = list(data)

    # remove course from list if nobody is taking it
    zero_exams = []
    for course in courses:
        # this sum is the number of people taking this course
        if sum(data[course]) == 0:
            zero_exams.append(course)
    
    # the actual removing of classes which nobody is taking
    data = data.drop(columns=zero_exams)
    for course in zero_exams:
        courses.remove(course)

    return (data, courses)

def get_departments(courses):
    """
    courses: a list of all courses formatted LLNNN
             (LL is department code)
    returns: a dictionary of departments where the 
             keys are LL and the values are lists of
             the courses in the departments
    """
    departments = {}
    for course in courses:
        prefix = course[:2]
        if prefix not in departments:
            departments[prefix] = [course]
        else:
            departments[prefix].append(course)

    return departments

def split_departments(data, departments, courses, definite_splits=[], restricted_splits=[], infer_splits=False):
    """
    data: pandas dataframe of course enrollments
    departments: dictionary of departments and courses in departments
    courses: list of all courses
    definite_splits: list of courses which should definitely be split

    returns None
    """

    split_departments = []
    
    # complete the definite splits
    for department in definite_splits:
        split_departments.append(department)

    if infer_splits:
        # complete inferred splits
        for department in departments:
            if department in definite_splits:
                continue
            block_1 = departments[department].copy()
            block_2 = []
            total_conflicts = count_conflicts(data, departments, block_1) + count_conflicts(data, departments, block_2)
            i = 0
            while i < len(block_1):
                block_2.append(block_1.pop(i))
                current_conflicts = count_conflicts(data, departments, block_1) + count_conflicts(data, departments, block_2)
                if current_conflicts != total_conflicts:
                    block_1.insert(i, block_2.pop())
                i += 1
            if len(block_2) == 0 or len(block_1) == 0:
                if department not in restricted_splits:
                    split_departments.append(department)

    # perform actual splitting in departments dictionary
    for department in split_departments:
        # split the department in the department dictionary
        departments[department+"-1"] = departments.pop(department)
        departments[department+"-2"] = []

        do_swapping(data, departments, departments[department+"-1"], departments[department+"-2"])

    return split_departments

def do_swapping(data, departments, section_1, section_2):
    """
    section_1: list of courses in the first department exam block
    section_2: list of courses in the second department exam block
    
    returns: None
    """

    sections = [section_1, section_2]

    conflicts_reduced = True
    total_conflicts = count_conflicts(data, departments, sections[0]) + count_conflicts(data, departments, sections[1])

    while conflicts_reduced:
        conflicts_reduced = False
        for section_index in (0, 1):
            # try swapping each course from one section to the other
            i = 0
            while i < len(sections[section_index]):
                # remove course from current section
                course = sections[section_index].pop(i)

                # move course to other section
                sections[~section_index].append(course)

                # move course back to current section if conflicts not reduced
                current_conflicts = count_conflicts(data, departments, sections[0]) + count_conflicts(data, departments, sections[1])
                if current_conflicts >= total_conflicts:
                    sections[section_index].insert(i, sections[~section_index].pop())
                else:
                    conflicts_reduced = True
                    total_conflicts = current_conflicts

                i += 1

    return None

def count_conflicts(data, departments, course_list):
    """
    data: pandas dataframe with rows of students taking columns of courses
    course_list: list of courses which share a given exam time

    returns: total conflicts in course_list
    """

    if len(course_list) == 0:
        return 0
    
    # if these are departments, not courses
    if course_list[0] in departments:         
        tmp = []
        for department in course_list:
            tmp += departments[department]
        course_list = tmp

    if len(course_list) == 0:
        return 0

    if len(course_list) == 1:
        return sum(data[course_list[0]] > 1)

    return sum(sum([data[i] for i in course_list]) > 1)
    
def assign_exam_blocks(data, departments, splitted_departments, number_days):
    """
    data: a pandas dataframe of the courses for each student
    departments: dictionary of departments and courses
    number_days: an integer for the number of days for exams

    returns: 2 dimensional list of departments
    """
    
    # create two exam blocks per day
    exam_blocks = [[] for i in range(2*number_days)]

    # fill exam_blocks with departments
    i = 0
    department_list = list(departments)
    index = np.random.permutation(np.arange(len(department_list)))
    for j in range(len(department_list)):
        department = department_list[index[j]]
        exam_blocks[i%(2*number_days)].append(department)
        i += 1

    total_conflicts = sum([count_conflicts(data, departments, block) for block in exam_blocks])
    conflicts_reduced = True
    while conflicts_reduced:
        print(total_conflicts)
        conflicts_reduced = False

        # do swapping between departments in exam blocks
        for i in range(len(exam_blocks)-1):
            for j in range(i+1, len(exam_blocks)):
                do_swapping(data, departments, exam_blocks[i], exam_blocks[j])
        
        # do swapping between the two blocks of split departments
        conflicts_reduced_2 = True
        while conflicts_reduced_2:
            current_conflicts = sum([count_conflicts(data, departments, block) for block in exam_blocks])
            conflicts_reduced_2 = False
            for department in splitted_departments:
                courses_1 = departments[department+"-1"]
                courses_2 = departments[department+"-2"]
                i = 0
                while i < len(courses_1):
                    courses_2.append(courses_1.pop(i))
                    tmp_conflicts = sum([count_conflicts(data, departments, block) for block in exam_blocks])
                    if tmp_conflicts >= current_conflicts:
                        courses_1.insert(i, courses_2.pop())
                    else:
                        conflicts_reduced_2 = True
                    i += 1
                
                i = 0
                while i < len(courses_2):
                    courses_1.append(courses_2.pop(i))
                    tmp_conflicts = sum([count_conflicts(data, departments, block) for block in exam_blocks])
                    if tmp_conflicts >= current_conflicts:
                        courses_2.insert(i, courses_1.pop())
                    else:
                        conflicts_reduced_2 = True
                    i += 1


        current_conflicts = sum([count_conflicts(data, departments, block) for block in exam_blocks])
        if current_conflicts < total_conflicts:
            total_conflicts = current_conflicts
            conflicts_reduced = True

    return exam_blocks

if __name__=='__main__':
    main()
