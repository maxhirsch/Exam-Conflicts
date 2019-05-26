import random
import itertools
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def place_reading_blocks_arrays(data, bins, num_read_blocks):
    data["read"] = 0
    for i in range(num_read_blocks):
        conflicts = [count_conflicts_arrays(data, bins[:j] + [["read"]] + bins[j:]) for j in range(len(bins)+1)]
        min_conflicts = min(conflicts)
        min_conflicts_index = conflicts.index(min_conflicts)
        bins = bins[:min_conflicts_index] + [["read"]] + bins[min_conflicts_index:]
    return bins

def get_minimum_bins(data, exams, num_exam_blocks, bins=None, rand=True):
    if bins is None:
        bins = [[] for i in range(num_exam_blocks)]
        if rand:
            # assign exams to random bins
            for i in range(len(exams)):
                bins[random.randint(0, num_exam_blocks-1)].append(exams[i])
        else:
            exams_per_bin = len(exams) // num_exam_blocks
            leftovers = len(exams) % num_exam_blocks
            for b in bins:
                for i in range(exams_per_bin):
                    b.append(exams.pop())
                if leftovers > 0:
                    b.append(exams.pop())
                    leftovers -= 1

    change_made = True
    while change_made: # (while number of conflicts still reduced)

        change_made = False
        # move backwards
        for i in range(len(bins)):
            j = 0
            while j < len(bins[i]):
                overlap_total, consecutive_total, bin_total = count_conflicts_arrays(data, bins)
                exam = bins[i][j]
                old_overlap = get_overlap(data, bins[i]) + get_overlap(data, bins[(i-1)%len(bins)])
                # get_consecutive returns the number of students with exams during bin k, k+1, and k+2
                old_consecutive = sum([get_consecutive(data, bins, k%len(bins)) for k in range(i-3, i+1)])
                
                # now move exam
                bins[i] = bins[i][:j] + bins[i][j+1:]
                bins[(i-1)%len(bins)].append(exam)
                
                new_overlap = get_overlap(data, bins[i]) + get_overlap(data, bins[(i-1)%len(bins)])
                # get_consecutive returns the number of students with exams during bin k, k+1, and k+2
                new_consecutive = sum([get_consecutive(data, bins, k%len(bins)) for k in range(i-3, i+1)])

                if old_overlap + old_consecutive <= new_overlap + new_consecutive: # if moving exam was bad
                    # move back
                    bins[i] = bins[i][:j] + [bins[(i-1)%len(bins)].pop()] + bins[i][j:]
                    j += 1
                else:
                    change_made = True
        
        # move forwards
        for i in range(len(bins)):
            j = 0
            while j < len(bins[i]):
                overlap_total, consecutive_total, bin_total = count_conflicts_arrays(data, bins)
                exam = bins[i][j]
                old_overlap = get_overlap(data, bins[i]) + get_overlap(data, bins[(i+1)%len(bins)])
                # get_consecutive returns the number of students with exams during bin k, k+1, and k+2
                old_consecutive = sum([get_consecutive(data, bins, k%len(bins)) for k in range(i-2, i+2)])
                
                # now move exam
                bins[i] = bins[i][:j] + bins[i][j+1:]
                bins[(i+1)%len(bins)].append(exam)
                
                new_overlap = get_overlap(data, bins[i]) + get_overlap(data, bins[(i+1)%len(bins)])
                # get_consecutive returns the number of students with exams during bin k, k+1, and k+2
                new_consecutive = sum([get_consecutive(data, bins, k%len(bins)) for k in range(i-2, i+2)])

                if old_overlap + old_consecutive <= new_overlap + new_consecutive: # if moving exam was bad
                    # move back
                    bins[i] = bins[i][:j] + [bins[(i+1)%len(bins)].pop()] + bins[i][j:]
                    j += 1
                else:
                    change_made = True
    
    return bins

def get_overlap(data, b):
    if len(b) == 0:
        return 0
    if len(b) == 1:
        return sum(data[b[0]] > 1)
    return sum(sum([data[i] for i in b]) > 1)

def get_consecutive(data, bins, i):
    if len(bins) - i < 3 or len(bins[i]) == 0 or len(bins[i+1]) == 0 or len(bins[i+2]) == 0:
        return 0
    has_exam = []
    for k in range(3):
        has_exam.append(sum([data[j] for j in bins[i+k]]) > 0)
    return sum(has_exam[0] & has_exam[1] & has_exam[2])

def count_conflicts_arrays(data, bins):
    overlap_total = 0
    consecutive_total = 0
    bin_conflicts = [0 for i in range(len(bins))]
    for i in range(len(bins)):
        current_bin = bins[i]
        overlap = get_overlap(data, current_bin)
        consecutive = get_consecutive(data, bins, i)
        bin_conflicts[i] = overlap + consecutive
        overlap_total += overlap
        consecutive_total += consecutive

    return (overlap_total, consecutive_total, bin_conflicts)


def get_minimum_bins_by_department(data, exams, num_exam_blocks, bins, departments):
    change_made = True
    while change_made: # (while number of conflicts still reduced)

        change_made = False
        # departments is a dictionary with keys equal to department
        # and value equal to a list of indices in which we can put
        # classes from that department
        
        for i in range(len(bins)):
            j = 0
            while j != len(bins[i]):
                e = bins[i][j]
                depot = e[:2]
                possible_bin_indices = departments[depot]
                min_conflicts = None
                min_conflicts_index = None
                bins[i] = bins[i][:j] + bins[i][j+1:]
                for index in possible_bin_indices:
                    # move e to bin[index],
                    # count conflicts,
                    # keep track of minimum conflicts index,
                    # move to minimum conflicts index,
                    # if this index != i, change_made = True
                    bins[index].append(e)
                    overlap_conflicts = sum([get_overlap(data, b) for b in bins])
                    consecutive_conflicts = sum([get_consecutive(data, bins, i) for i in range(len(bins))])
                    total_conflicts = overlap_conflicts + consecutive_conflicts
                    if min_conflicts is None or total_conflicts < min_conflicts:
                        min_conflicts = total_conflicts
                        min_conflicts_index = index
                    bins[index].pop()
                if min_conflicts_index == i:
                    bins[i] = bins[i][:j] + [e] + bins[i][j:]
                    j += 1
                else:
                    change_made = True
                    bins[min_conflicts_index].append(e)
    return bins

def probabilistic_least_overlapping_departments(department_df, department_list, num_blocks):
    department_list = department_list.copy()
    department_df = department_df.copy()
    
    combos = [list(itertools.combinations(department_list, i)) for i in range(1, 5)]
    
    #combos = combos2 + combos3 + combos4 #+ combos5 + combos6
    combo_conflicts = []
    combo_sort_indices = []
    department_df[department_df > 0] = 1 
    for i in range(1, 5):
        conflicts = []
        for c in combos[i-1]:
            array = department_df[c[0]]
            for j in range(1, len(c)):
                array = array + department_df[c[j]]
            conflicts.append((sum(array > 1), -len(c)))
        conflicts = np.array(conflicts, dtype=[('x', 'int64'), ('y', 'int64')])
        indices = np.argsort(conflicts, order=('x'))#, 'y'))
        combo_conflicts.append(conflicts)
        combo_sort_indices.append(indices)
    
    single_classes = set()
    groupings = []
    n = len(department_list)
    k = num_blocks
    position_index = [0 for i in range(len(combos))] # keep track of position in arrays
    while k > 0:
        min_conflicts = None
        min_conflicts_combo = None
        # we can check indices n//k or greater 
        for i in range(n//k-1, len(combos)):
            if position_index[i] >= len(combos[i]):
                continue
            # make the next possible combination distinct from current class set
            while len(set(combos[i][combo_sort_indices[i][position_index[i]]]) & single_classes) > 0: # intersection
                position_index[i] += 1 
                if position_index[i] >= len(combos[i]):
                    break
            if position_index[i] >= len(combos[i]):
                continue
            # check to see if it could possibly be the minimum value
            if min_conflicts is None or combo_conflicts[i][combo_sort_indices[i][position_index[i]]][0] <= min_conflicts:
                min_conflicts = combo_conflicts[i][combo_sort_indices[i][position_index[i]]][0]
                min_conflicts_combo = combos[i][combo_sort_indices[i][position_index[i]]]
        # add min_conflict_combo to single_classes set and groupings list
        if min_conflicts_combo is None:
            return None
        single_classes.update(min_conflicts_combo)
        groupings.append(list(min_conflicts_combo))
        k -= 1
        n -= len(min_conflicts_combo)
    
    return groupings

def do_swaps(department_df, blocks):

    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)): 
            j = 0
            while j != len(blocks[i]):
                exam = blocks[i][j]
                i_conflicts = get_overlap(department_df, blocks[i]) #+ get_consecutive(department_df, blocks, i)
                blocks[i] = blocks[i][:j] + blocks[i][j+1:]
                for k in range(len(blocks)):
                    if k == i:
                        continue
                    m = 0
                    while m != len(blocks[k]):
                        exam2 = blocks[k][m]
                        k_conflicts = get_overlap(department_df, blocks[k]) #+ get_consecutive(department_df, blocks, k)
                        blocks[k] = blocks[k][:m] + blocks[k][m+1:]
                        # try swapping blocks[k][m] and blocks[i][j]
                        blocks[i].append(exam2)
                        blocks[k].append(exam)
                        new_i_conflicts = get_overlap(department_df, blocks[i]) #+ get_consecutive(department_df, blocks, i)
                        new_k_conflicts = get_overlap(department_df, blocks[k]) #+ get_consecutive(department_df, blocks, k)
                        if new_i_conflicts + new_k_conflicts < i_conflicts + k_conflicts: # keep switch
                            changed = True
                            break
                        else:
                            blocks[k].pop()
                            blocks[i].pop()
                            blocks[k] = blocks[k][:m] + [exam2] + blocks[k][m:]
                            m += 1
                        if m == len(blocks[k]):
                            blocks[k].append(exam)
                            new_i_conflicts = get_overlap(department_df, blocks[i]) #+ get_consecutive(department_df, blocks, i)
                            new_k_conflicts = get_overlap(department_df, blocks[k]) #+ get_consecutive(department_df, blocks, k)
                            if new_i_conflicts + new_k_conflicts < i_conflicts + k_conflicts:
                                changed = True
                                break
                            else:
                                blocks[k].pop()
                            
                    if changed:
                        break
                if changed:
                    break
                else:
                    blocks[i] = blocks[i][:j] + [exam] + blocks[i][j:]
                    j += 1
            if changed:
                break

    return blocks



def get_minimum_overlapping_departments(data, exams, num_blocks, num_read):
    # put each department in a dictionary 
    # and record which students have classes
    # in that department (signified by department[d] > 0)
    
    exams = exams.copy()
    departments = {}
    courses = {}
    for e in exams:
        if e[:2] not in departments:
            departments[e[:2]] = data[e].copy()
        else:
            departments[e[:2]] += data[e].copy()

    department_list = list(departments)
    department_df = pd.DataFrame(departments)
    

    courses = split_departments(department_df, department_list, data, exams)
    """
    # Trimester 1
    departments = [["PH-2", "CH-1"], ["BI"], [], ["SS", "CN", "FR", "JA", "LA", "SP"], ["EE", "CS", "IE", "MS"], [], ["MA-1"], ["PH-1", "CH-2"], [], ["AS", "EN"], ["MA-2"]]
    do_swaps(department_df, departments)
    print(departments, courses)
    x = [get_overlap(department_df, b) for b in departments]
    print(x, sum(x))
    """
    """
    # Trimester 2
    departments = [["PH-2", "CH-1"], ["BI"], ["HU", "SS", "CN", "FR", "JA", "LA", "SP"], ["EE", "CS", "IE", "MS"], ["MA-1"], ["PH-1", "CH-2"], ["AS", "EN"], ["MA-2"]]
    do_swaps(department_df, departments)
    print(departments, courses)
    x = [get_overlap(department_df, b) for b in departments]
    print(x, sum(x))
    """
    #"""
    # Trimester 3
    departments = [["PH-2", "PH-1", "CH-1", "CH-2"], ["BI-1", "BI-2"], [], ["MS", "EE", "CS", "IE"], ["CN", "FR", "JA", "LA", "SP", "HU", "SS"], [], ["MA-1", "MA-2"], ["AS", "EN"]]
    do_swaps(department_df, departments) # swap departments around
    print(departments, courses)
    x = [get_overlap(department_df, b) for b in departments]
    print(x, sum(x))#, count_conflicts_arrays(department_df, departments))
    #"""
    #sys.exit(0)

    plod = probabilistic_least_overlapping_departments(department_df, department_list, num_blocks)
    
    min_overlap, min_consecutive, min_array = count_conflicts_arrays(department_df, plod)
    reduced = True
    while reduced:
        reduced = False
        bins = []
        #plod = get_minimum_bins(department_df.copy(), department_list, num_blocks, plod)
        plod = do_swaps(department_df.copy(), plod)

        for i in range(len(plod)):
            bins.append([])
            for j in range(len(plod[i])):
                if plod[i][j][:4] in courses:
                    bins[-1] += courses[plod[i][j][:4]] # courses has changed after the first iteration
                else:
                    bins[-1] += courses[plod[i][j][:2]]

        department_bins = {}
        for i in range(len(bins)):
            b = bins[i]
            for e in b:
                depot = e[:2]
                if depot not in department_bins:
                    department_bins[depot] = [i]
                elif i not in department_bins[depot]:
                    department_bins[depot].append(i)

        get_minimum_bins_by_department(data, exams, num_blocks, bins, department_bins)

        courses = {} # we need to update courses and the department_df

        for i in range(len(plod)):
            for j in range(len(plod[i])):
                first = True
                found = False
                split_in_same_container = (plod[i][j][:2] + "-1" in plod[i][j:]) and (plod[i][j][:2] + "-2" in plod[i][j:]) 
                if split_in_same_container: # we used slice to skip only first split department in above line
                    department_df[plod[i][j]].values[:] = 0
                    found = True
                    courses[plod[i][j]] = []
                    continue
                for k in range(len(bins[i])):
                    if bins[i][k][:2] == plod[i][j][:2]:
                        if first:
                            department_df[plod[i][j]] = data[bins[i][k]].copy() 
                            first = False
                            found = True
                            courses[plod[i][j]] = [bins[i][k]]
                        else:
                            department_df[plod[i][j]] += data[bins[i][k]].copy()
                            courses[plod[i][j]].append(bins[i][k])

                if not found:
                    department_df[plod[i][j]].values[:] = 0
                    courses[plod[i][j]] = []
                    
        current_overlap, current_consecutive, current_array = count_conflicts_arrays(data, bins)
        equivalent = [current_array[i] == min_array[i] for i in range(len(current_array))]
        if False not in equivalent:
            reduced = True
            min_overlap = current_overlap
            min_consecutive = current_consecutive
            min_array = current_array
    
    bins = place_reading_blocks_arrays(data, bins, num_read)
    print(plod)
    return bins

# separate classes within department into two separate bins
# one bin contains classes that cannot be taken at the same time

def split_departments(department_df, department_list, data, exams):
    d_list_copy = department_list.copy()
    courses = {}
    for d in d_list_copy:

        bin1 = []
        bin2 = []
        # move all exams into bin1
        for e in exams:
            if e[:2] == d:
                bin1.append(e)
        
        """
        if d not in ("PH", "CH", "MA", "BI"):
            courses[d] = bin1
            continue
        """

        # for each exam in bin1, move exam to bin2 if it doesn't reduce conflicts
        i = 0
        while i < len(bin1):
            e = bin1[i]
            move = True
            for e2 in bin1:
                #print(e, e2, sum((data[e] > 0) & (data[e2] > 0)))
                if e is not e2 and sum((data[e] > 0) & (data[e2] > 0)) > 0:
                    move = False
                    break
            if move:
                for e2 in bin2:
                    if sum((data[e] > 0) & (data[e2] > 0)) > 0:
                        move = False
                if move:
                    bin1.remove(e)
                    bin2.append(e)
                else:
                    i += 1
            else:
                i += 1
        
        # if not part of code
        #if d == 'MA' or d == 'PH' or d == 'BI' or d == 'CH':
        bin1, bin2 = get_minimum_bins(data, bin1+bin2, 2, [bin1, bin2])

        if (len(bin1) > 0 and len(bin2) > 0) or d in ('MA', 'PH', 'BI', 'CH'): # then split department classes
            department_list.remove(d)
            department_list.append(d + "-1")
            department_list.append(d + "-2")
            department_df[d + "-1"] = sum([data[c] for c in bin1])
            department_df[d + "-2"] = sum([data[c] for c in bin2])
            courses[d + "-1"] = bin1
            courses[d + "-2"] = bin2
        else:
            if len(bin1) > 0:
                courses[d] = bin1
            else:
                courses[d] = bin2

    return courses


if __name__=='__main__':

    np.random.seed(42)

    data = pd.read_csv("classes_2018.csv")
    data = data[data.Trimester == "T3"]

    exams = list(data)
    exams.remove("Student ID")
    exams.remove("Trimester")
    zero_exams = []
    for e in exams:
        if sum(data[e]) == 0:
            zero_exams.append(e)
    data = data.drop(columns=zero_exams)
    for e in zero_exams:
        exams.remove(e)
    
    mod = get_minimum_overlapping_departments(data, exams, 8, 0)
    print(mod, count_conflicts_arrays(data, mod))
    
