import sys
import pandas as pd

data = pd.read_csv("schedule_u.csv")
classes = []
with open("whitelist.txt", "r") as infile:
    for line in infile:
        classes.append(line.strip())

new_csv_16 = pd.DataFrame(columns=["Student ID", "Trimester"] + classes)
new_csv_17 = pd.DataFrame(columns=["Student ID", "Trimester"] + classes)
new_csv_18 = pd.DataFrame(columns=["Student ID", "Trimester"] + classes)
heading = list(new_csv_16)


l = len(data[data["schedule_year"] == 2016])
for index, row in data[data["schedule_year"] == 2016].iterrows():
    print(index, '/', l, end='\r')
    student = row["student_number"]
    tri = row["term"]
    if len(new_csv_16[(new_csv_16["Student ID"] == student) & (new_csv_16["Trimester"] == tri)]) == 0:
        df = pd.DataFrame([[student, tri] + [0 for i in range(len(classes))]], columns=["Student ID", "Trimester"] + classes)
        new_csv_16 = new_csv_16.append(df)
    
    course = row["course_title"][:6].strip()
    if course in heading:
        new_csv_16.loc[(new_csv_16["Student ID"] == student) & (new_csv_16["Trimester"] == tri), course] += 1 

new_csv_16.to_csv('classes_2016.csv', index=False)
print(new_csv_16.head())

l = len(data[data["schedule_year"] == 2017])
for index, row in data[data["schedule_year"] == 2017].iterrows():
    print(index, '/', l, end='\r')
    student = row["student_number"]
    tri = row["term"]
    if len(new_csv_17[(new_csv_17["Student ID"] == student) & (new_csv_17["Trimester"] == tri)]) == 0:
        df = pd.DataFrame([[student, tri] + [0 for i in range(len(classes))]], columns=["Student ID", "Trimester"] + classes)
        new_csv_17 = new_csv_17.append(df)
    
    course = row["course_title"][:6].strip()
    if course in heading:
        new_csv_17.loc[(new_csv_17["Student ID"] == student) & (new_csv_17["Trimester"] == tri), course] += 1 

new_csv_17.to_csv('classes_2017.csv', index=False)
print(new_csv_17.head())

l = len(data[data["schedule_year"] == 2018])
for index, row in data[data["schedule_year"] == 2018].iterrows():
    print(index, '/', l, end='\r')
    student = row["student_number"]
    tri = row["term"]
    if len(new_csv_18[(new_csv_18["Student ID"] == student) & (new_csv_18["Trimester"] == tri)]) == 0:
        df = pd.DataFrame([[student, tri] + [0 for i in range(len(classes))]], columns=["Student ID", "Trimester"] + classes)
        new_csv_18 = new_csv_18.append(df)
    
    course = row["course_title"][:6].strip()
    if course in heading:
        new_csv_18.loc[(new_csv_18["Student ID"] == student) & (new_csv_18["Trimester"] == tri), course] += 1 

new_csv_18.to_csv('classes_2018.csv', index=False)
print(new_csv_18.head())
