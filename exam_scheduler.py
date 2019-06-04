import pandas as pd
import numpy as np

def get_exam_schedule(filename, trimester, definite_splits=[], restricted_splits=[], infer_splits=True, 
        number_exam_days=4, number_random_trials=5, random_seed=42):
    """
    Creates an exam schedule with few conflicts.

    Parameters:
    -----------
    filename (str): The name of the data file
    trimester (str): The name of the trimester ("T1", "T2", "T3")
    definite_splits (list): Courses which should definitely be split into multiple exam blocks
    restricted_splits (list): Courses which should definitely not be split into multiple exam blocks
    infer_splits (bool): If True, departments not specified in definite_splits or restricted_splits 
        are split into two departments if every course in the department conflicts with at least one other 
        course in the department. If False, this operation will not be performed
    number_exam_days (int): The number of days for exams
    number_random_trials (int): The number of initial random assignments for the exam schedule to 
        try to optimize
    random_seed (int): A random seed for the random number generator


    returns (tuple): the exam blocks with department codes in lists, classes in the departments as a dict,
        and the total number of conflicts in the schedule
    """

    # seed any randomness for reproducibility
    np.random.seed(random_seed)

    # get data
    data = load_data(filename, trimester)
    courses = list(data)

    # get departments
    departments = get_departments(courses)
    
    # split departments ("splitted" b/c naming collisions...)
    splitted_departments = split_departments(data, departments, courses, definite_splits=definite_splits, 
            restricted_splits=restricted_splits, infer_splits=infer_splits)

    # assign departments to exam blocks
    exam_blocks = assign_exam_blocks(data, departments, splitted_departments, number_exam_days=number_exam_days)
    total_conflicts = sum([count_conflicts(data, departments, block) for block in exam_blocks])

    # reassign departments to exam blocks and keep assignment with fewest conflicts
    for trial in range(number_random_trials-1):
        current_exam_blocks = assign_exam_blocks(data, departments, splitted_departments, number_exam_days=4)
        current_conflicts = sum([count_conflicts(data, departments, block) for block in current_exam_blocks])
        if current_conflicts < total_conflicts:
            exam_blocks = current_exam_blocks
            total_conflicts = current_conflicts

    return (exam_blocks, departments, total_conflicts)

def load_data(filename, trimester):
    """
    Loads the student exam data into a pandas DataFrame.

    Parameters:
    -----------
    filename (str): The name of the data file
    trimester (str): The name of the trimester ("T1", "T2", "T3")


    returns (pandas.DataFrame): Student course enrollments
    """

    # load data
    data = pd.read_csv(filename)

    # filter data to desired trimester
    data = data[data.Trimester == trimester]

    # drop columns we won't use
    data = data.drop(columns=["Student ID", "Trimester"])
    
    # remove course from course list if nobody is taking it
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

    return data

def get_departments(courses):
    """
    Creates a dictionary of departments and courses in those departments.

    Parameters:
    -----------
    courses (list): All courses formatted LLNNN (LL is a 2 character department code)
    

    returns (dict): Departments where the keys are LL and the values are lists of
        the courses in the departments
    """
    
    departments = {}
    for course in courses:
        prefix = course[:2]
        # either add department prefix and course to dictionary or
        # add course to department list in the dictionary
        if prefix not in departments:
            departments[prefix] = [course]
        else:
            departments[prefix].append(course)

    return departments

def split_departments(data, departments, courses, definite_splits=[], restricted_splits=[], infer_splits=False):
    """
    Splits departments into multiple exam blocks.

    Parameters:
    -----------
    data (pandas.DataFrame): Course enrollments data
    departments (dict): Departments (str key) and courses in departments (list value)
    courses (list): All courses
    definite_splits (list): Courses which should definitely be split into multiple exam blocks
    restricted_splits (list): Courses which should definitely not be split into multiple exam blocks
    infer_splits (bool): If True, departments not specified in definite_splits or restricted_splits 
        are split into two departments if every course in the department conflicts with at least one other 
        course in the department. If False, this operation will not be performed


    returns (list): Departments which were split
    """

    split_departments = []
    
    # complete the definite splits
    for department in definite_splits:
        split_departments.append(department)

    # infer splits according to rule in function docstring
    if infer_splits:
        # try splitting each department
        for department in departments:
            # definite_splits and restricted splits are handled separately
            if department in definite_splits or department in restricted_splits:
                continue
            # create two exam blocks to put this department's courses into
            block_1 = departments[department].copy()
            block_2 = []
            # keep track of conflicts for applying the rule (the while loop)
            total_conflicts = count_conflicts(data, departments, block_1) + count_conflicts(data, departments, block_2)
            
            # try putting each class into block_2 from block_1; if the new number of conflicts
            i = 0
            while i < len(block_1):
                block_2.append(block_1.pop(i))

                # if the new number of conflicts is the same as before, either this class did not conflict with anything or
                # it has the same number of conflicts regardless of the exam block
                # (but the first course to move to block_2 without reducing conflicts must not have conflicted with anything)
                
                current_conflicts = count_conflicts(data, departments, block_1) + count_conflicts(data, departments, block_2)
                if current_conflicts != total_conflicts:
                    # move back to block_1 since the number of conflicts changed
                    block_1.insert(i, block_2.pop())
                i += 1

            # if this is true, then all courses in the department conflicted with every other course in the department
            # so the department should split according to the rule in the docstring
            if len(block_2) == 0:
                split_departments.append(department)

    # perform actual splitting in departments dictionary
    for department in split_departments:
        departments[department+"-1"] = departments.pop(department)
        departments[department+"-2"] = []
        # balance the two exam blocks for this department to reduce conflicts
        do_swapping(data, departments, departments[department+"-1"], departments[department+"-2"])

    return split_departments

def do_swapping(data, departments, section_1, section_2):
    """
    Swap departments between exam blocks or courses between split department blocks.

    Parameters:
    -----------
    data (pandas.DataFrame): Course enrollments data
    departments (dict): Departments (str key) and courses in departments (list value)
    section_1 (list): Departments in one exam block or courses in the first department exam block
    section_2 (list): Departments in another exam block or courses in the second department exam block
    
    returns (NoneType): None
    """

    sections = [section_1, section_2]
    conflicts_reduced = True
    total_conflicts = count_conflicts(data, departments, sections[0]) + count_conflicts(data, departments, sections[1])

    # keep swapping if improvements are made
    while conflicts_reduced:
        conflicts_reduced = False
        # this allows us to swap from section_1 to section_2 then section_2 to section_1
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

def count_conflicts(data, departments, section):
    """
    Count the number of conflicts in the section (either exam block or department).

    Parameters:
    -----------
    data (pandas.DataFrame): Course enrollments data
    section (list): Departments which share an exam time or courses which share a department


    returns (int): Total conflicts in section
    """

    # no possible conflicts if empty section
    if len(section) == 0:
        return 0
    
    # if these are departments, not courses, then get list of courses in these departments
    if section[0] in departments:
        tmp = []
        for department in section:
            tmp += departments[department]
        section = tmp

    # if after getting courses from departments, there were no courses, then no conflicts
    if len(section) == 0:
        return 0

    # if only one course, then there can't be conflicts
    if len(section) == 1:
        return 0

    # add students enrollments for these courses together; the total number of students
    # with more than one enrollment in these courses is the total number of conflicts
    return sum(sum([data[i] for i in section]) > 1)
    
def assign_exam_blocks(data, departments, splitted_departments, number_exam_days):
    """
    Assign departments to exam blocks and optimize this schedule to reduce conflicts.

    data (pandas.DataFrame): Course enrollments data
    departments (dict): Departments (str key) and courses in departments (list value)
    number_exam_days (int): The number of days for exams


    returns (list): Departments for each exam block
    """
    
    # create two exam blocks per day
    exam_blocks = [[] for i in range(2*number_exam_days)]

    # sequentially fill exam_blocks with departments in random order
    i = 0
    department_list = list(departments)
    index = np.random.permutation(np.arange(len(department_list)))
    for j in range(len(department_list)):
        department = department_list[index[j]]
        exam_blocks[i%(2*number_exam_days)].append(department)
        i += 1

    # swap exam blocks until this swap method can no longer reduce conflicts
    total_conflicts = sum([count_conflicts(data, departments, block) for block in exam_blocks])
    conflicts_reduced = True
    while conflicts_reduced:
        conflicts_reduced = False

        # do swapping between departments in exam block i and exam block j
        for i in range(len(exam_blocks)-1):
            for j in range(i+1, len(exam_blocks)):
                do_swapping(data, departments, exam_blocks[i], exam_blocks[j])
        
        # do swapping between the two blocks of split departments to try to minimize
        # conflicts on a course basis
        course_conflicts_reduced = True
        while course_conflicts_reduced:
            current_conflicts = sum([count_conflicts(data, departments, block) for block in exam_blocks])
            course_conflicts_reduced = False
            # do this for every splitted department
            for department in splitted_departments:
                courses = [departments[department+"-1"], departments[department+"-2"]]
                # this allows us to swap from first department exam block to other department exam block
                for course_index in (0, 1):
                    # swap any course from one department section to the other if it reduces conflicts
                    i = 0
                    while i < len(courses[course_index]):
                        courses[~course_index].append(courses[course_index].pop(i))
                        tmp_conflicts = sum([count_conflicts(data, departments, block) for block in exam_blocks])
                        if tmp_conflicts >= current_conflicts:
                            courses[course_index].insert(i, courses[~course_index].pop())
                        else:
                            course_conflicts_reduced = True
                        i += 1
        
        current_conflicts = sum([count_conflicts(data, departments, block) for block in exam_blocks])
        if current_conflicts < total_conflicts:
            total_conflicts = current_conflicts
            conflicts_reduced = True

    return exam_blocks

def print_exam_schedule(exam_schedule):
    """
    Prints the exam schedule in a prettier format.

    Parameters:
    -----------
    exam_schedule (2D list): Departments for each exam block


    returns (NoneType): None
    """

    for i in range(len(exam_schedule)):
        print("Block " + str(i+1) + ": [" + ', '.join(exam_schedule[i]) + "]")

    return None

def print_departments(departments):
    """
    Prints the department courses in a prettier format.

    Parameters:
    -----------
    departments (dict): Departments (str key) and courses in departments (list value)


    returns (NoneType): None
    """

    for department, courses in departments.items():
        print(department + ": [" + ', '.join(courses) + "]")

    return None

if __name__=='__main__':
    exam_blocks, departments, total_conflicts = get_exam_schedule("classes_2018.csv", "T2", 
            definite_splits=["MA", "CH", "PH", "BI"], number_random_trials=1)

    print_exam_schedule(exam_blocks)
    print()
    print_departments(departments)
    print()
    print("Total conflicts:", total_conflicts)
