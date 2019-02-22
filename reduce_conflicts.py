"""
Goal:

    Make GUI which allows input of number of exam days
    and minimizes the number of exam conflicts
    
"""


import pandas as pd
import numpy as np

def count_conflicts(data, exam_order, trimester, consecutive=True, overlap=True):
    # filter dataframe to given trimester
    data = data[data.title == trimester]

    count = 0
    if consecutive:
        for index, row in data.iterrows():
            # index into row using exam order
            student_exam_sched = list(row[exam_order])
            # count number of 3-consecutive exams
            count += count_threes(student_exam_sched)
    
    if overlap:
        for index, row in data.iterrows():
            # index into row using exam order
            student_exam_sched = list(row[exam_order])
            # number of more than 1 exam in block
            count += count_overlaps(student_exam_sched)

    return count

def count_threes(array):
    i, j, k = 0, 1, 2

    count = 0
    while k < len(array):
        if array[i] > 0 and array[j] > 0 and array[k] > 0:
            count += 1
        i += 1
        j += 1
        k += 1

    return count

def count_overlaps(array):
    count = 0
    for i in array:
        if i > 1:
            count += 1 # if we want contribution of 2 conflicts given 3 exams in block, use i-1 instead of 1
    return count

def naive_minimization(data, exam_order, trimester, trials = 1000, seed = 42):
    """
    Simply randomly permute the exam order, keeping
    track of minimum conflict exam_order
    """
    np.random.seed(seed)

    min_conflicts = count_conflicts(data, exam_order, trimester)
    min_order = exam_order

    for i in range(trials):
        order = exam_order[np.random.permutation(len(exam_order))]
        num_conflicts = count_conflicts(data, order, trimester)
        if num_conflicts < min_conflicts:
            min_conflicts = num_conflicts
            min_order = order
        print(i/trials, min_conflicts)

    return (min_order, min_conflicts)

def greedy_minimization(data, exam_order, trimester, consecutive=True, overlap=True):
    exam_order = list(exam_order)
    # remove "read" from exam_order; we will add these back in later
    num_read = exam_order.count("read")
    for i in range(num_read):
        exam_order.remove("read")
    
    greedy_order = []

    while len(exam_order) > 0:

        conflicts = [count_conflicts(data, np.array(greedy_order + [e]), trimester, consecutive, overlap) for e in exam_order]
        min_conflicts = min(conflicts)
        min_conflicts_index = conflicts.index(min_conflicts)
        greedy_order.append(exam_order[min_conflicts_index])
        exam_order = exam_order[:min_conflicts_index] + exam_order[min_conflicts_index+1:]

    # add back "read" blocks
    for i in range(num_read):
        conflicts = [count_conflicts(data, np.array(greedy_order[:j] + ["read"] + greedy_order[j:]), trimester, consecutive, overlap) for j in range(len(greedy_order)+1)]
        min_conflicts = min(conflicts)
        min_conflicts_index = conflicts.index(min_conflicts)
        greedy_order = greedy_order[:min_conflicts_index] + ["read"] + greedy_order[min_conflicts_index:]
   
    return (greedy_order, count_conflicts(data, np.array(greedy_order), trimester, consecutive, overlap))

def move_to_makeup_block(data, exam_order, trimester, makeup_block_num=1):
    data = data[data.title == trimester]
    for index, row in data.iterrows():
        for i in range(len(exam_order)):
            if row[exam_order[i]] > 1:
                row[exam_order[i]] -= 1
                row["makeup" + str(makeup_block_num)] += 1
                break
    return
                
def pair_exams(exams, num_pairs):
    if num_pairs == 0:
        return [exams]
    pairings = []
    for i in range(len(exams)):
        for j in range(i+1, len(exams)):
            other_pairings = pair_exams(exams[i+1:j] + exams[j+1:], num_pairs-1)
            for p in other_pairings:
                pairings.append([exams[k] for k in range(i)] + [exams[i] + '_' + exams[j]] + p)
    return pairings

def add_pairs_to_dataframe(data, exams, num_pairs):
    if num_pairs == 0:
        return [exams]
    pairings = []
    for i in range(len(exams)):
        for j in range(i+1, len(exams)):
            other_pairings = pair_exams(exams[i+1:j] + exams[j+1:], num_pairs-1)
            for p in other_pairings:
                pairings.append([exams[k] for k in range(i)] + [exams[i] + '_' + exams[j]] + p)
                data[exams[i] + '_' + exams[j]] = data[exams[i]] + data[exams[j]]
    return pairings


if __name__=='__main__':

    data_2017_18 = pd.read_csv("Exam-Conflicts-2018.csv")

    exam_order_1 = np.array(["read", "adv_math_elec", "en400", "bio", "math", \
            "ee_elec_sci", "chem", "wl", "cs_elec_hum", "phys", \
            "as_ss"])
    exam_order_2 = np.array(["ee_elec_sci", "bio", "math", "read", "chem", \
            "wl", "cs_elec_hum", "phys", "as_ss", "adv_math_elec", "en400"])
    exam_order_3 = np.array(["ee_elec_sci", "en400_as_ss", "math", "cs_elec_hum", \
            "chem_phys", "adv_math_elec", "wl", "bio"])

    data_2017_18["ee_elec_sci"] = data_2017_18["ee"] + data_2017_18["elec_sci"]
    data_2017_18["cs_elec_hum"] = data_2017_18["cs"] + data_2017_18["elec_hum"]
    data_2017_18["as_ss"] = data_2017_18["as"] + data_2017_18["ss"]
    data_2017_18["en400_as_ss"] = data_2017_18["en400"] + data_2017_18["as_ss"]
    data_2017_18["chem_phys"] = data_2017_18["chem"] + data_2017_18["phys"]
    data_2017_18["read"] = 0 * data_2017_18["math"] # set to column of 0s
    data_2017_18["makeup1"] = data_2017_18["read"] # set to column of 0s
    data_2017_18["makeup2"] = data_2017_18["read"] # set to column of 0s

    num_blocks = 10

    """
    total = count_conflicts(data_2017_18, exam_order_3, "Trimester 3")

    print(total)
    print(exam_order_3)

    i = 1
    while "makeup" + str(i) in exam_order_1:
        move_to_makeup_block(data_2017_18, exam_order_1, "Trimester 1", i)
        i += 1

    print(greedy_minimization(data_2017_18, exam_order_3, "Trimester 3"))
    """

    exams = ["adv_math_elec", "en400", "bio", "math", "ee", "elec_sci", "chem", "wl", "cs", "elec_hum", "phys", "as", "ss"]
    paired_exams = []
    
    """
    for i in range(len(exams)//2+1):
        p = pair_exams(exams, i)
        print(len(p))
        paired_exams += p#pair_exams(exams, i)
    """
    paired_exams = pair_exams(exams, len(exams)-num_blocks)
    print(len(paired_exams))

    pairs_within_num_blocks = []
    for i in range(len(paired_exams)):
        if len(paired_exams[i]) <= num_blocks:
            pairs_within_num_blocks.append(np.array(paired_exams[i]))

    print(len(pairs_within_num_blocks))

    # add each pair of exam blocks to the dataframe
    data_pairings = add_pairs_to_dataframe(data_2017_18, exams, len(exams)//2)
    
    progress_total = len(pairs_within_num_blocks)
    # determine best pairing setup
    min_order, min_conflicts = greedy_minimization(data_2017_18, pairs_within_num_blocks[0], "Trimester 1", consecutive=False)
    for i in range(len(pairs_within_num_blocks)):
        print(i/progress_total)
        order, conflicts = greedy_minimization(data_2017_18, pairs_within_num_blocks[i], "Trimester 1", consecutive=False)
        if conflicts < min_conflicts:
            min_conflicts = conflicts
            min_order = order










