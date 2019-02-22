import sys
"""
Goal:

    Make GUI which allows input of number of exam days
    and minimizes the number of exam conflicts
    
"""


import pandas as pd
import numpy as np

def count_conflicts(data, exam_order, trimester, consecutive=True, overlap=True, minimum=None):
    # filter dataframe to given trimester
    data = data[data.title == trimester]

    count = 0
    if consecutive:
        for index, row in data.iterrows():
            # index into row using exam order
            student_exam_sched = list(row[exam_order])
            # count number of 3-consecutive exams
            count += count_threes(student_exam_sched)
            if minimum != None and count >= minimum:
                return -1
    
    if overlap:
        for index, row in data.iterrows():
            # index into row using exam order
            student_exam_sched = list(row[exam_order])
            # number of more than 1 exam in block
            count += count_overlaps(student_exam_sched)
            if minimum != None and count >= minimum:
                return -1

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
    #np.random.seed(seed)

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

def greedy_minimization(data, exam_order, trimester):
    exam_order = list(exam_order)

    # remove "read" from exam_order; we will add these back in later
    num_read = exam_order.count("read")
    for i in range(num_read):
        exam_order.remove("read")
    
    greedy_order = []
    
    while len(exam_order) > 0:
        conflicts = [count_conflicts(data, np.array(greedy_order + [e]), trimester, overlap=False) for e in exam_order]
        min_conflicts = min(conflicts)
        min_conflicts_index = conflicts.index(min_conflicts)
        greedy_order.append(exam_order[min_conflicts_index])
        exam_order = exam_order[:min_conflicts_index] + exam_order[min_conflicts_index+1:]

    # add back "read" blocks
    for i in range(num_read):
        conflicts = [count_conflicts(data, np.array(greedy_order[:j] + ["read"] + greedy_order[j:]), 
                trimester, overlap=False) for j in range(len(greedy_order)+1)]
        min_conflicts = min(conflicts)
        min_conflicts_index = conflicts.index(min_conflicts)
        greedy_order = greedy_order[:min_conflicts_index] + ["read"] + greedy_order[min_conflicts_index:]
    
    return (greedy_order, count_conflicts(data, np.array(greedy_order), trimester, overlap=False))

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

def sample(array, n_samples):
    if type(array) == list:
        array = np.array(array)
    return array[np.random.permutation(len(array))[:n_samples]]

def get_minimum(data, samples, trimester, consecutive=True, overlap=True):
    if len(samples) == 0:
        return None
    
    min_order, min_conflicts = greedy_minimization(data, samples[0], trimester)
    min_conflicts += count_conflicts(data, samples[0], trimester, consecutive=False)
    
    i = 1
    t = len(samples)
    
    for s in samples[1:]:
        print(i/t)
        conflicts = count_conflicts(data, s, trimester, consecutive=False, minimum=min_conflicts)
        if conflicts == -1:
            continue
        
        order, consec_conflicts = greedy_minimization(data, s, trimester)
        conflicts += consec_conflicts
        
        if conflicts < min_conflicts:
            min_order = order
            min_conflicts = conflicts
    
    return (min_order, min_conflicts)

if __name__=='__main__':

    np.random.seed(42)

    data_2017_18 = pd.read_csv("Exam-Conflicts-2018.csv")
    exams = ["adv_math_elec", "en400", "bio", "math", "ee", "elec_sci", "chem", "wl", "cs", "elec_hum", "phys", "as", "ss"]
    
    # add each pair of exam blocks to the dataframe
    data_pairings = add_pairs_to_dataframe(data_2017_18, exams, len(exams)//2)
    
    data_2017_18["read"] = 0 * data_2017_18["math"] # set to column of 0s

    num_blocks = 10
    num_reading = 1

    paired_exams = pair_exams(exams, len(exams)-num_blocks)
    pairs_sample = list(sample(paired_exams, 2))
    
    for i in range(len(pairs_sample)):
        for read_block in range(num_reading):
            pairs_sample[i] = np.append(pairs_sample[i], "read")
  

    
    min_order, min_conflicts = get_minimum(data_2017_18, pairs_sample, "Trimester 1")

    print(min_order, min_conflicts)

