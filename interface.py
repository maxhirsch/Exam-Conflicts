#!/usr/bin/python3
"""
Based on https://www.python-course.eu/tkinter_entry_widgets.php
"""
from tkinter import *
from tkinter.filedialog import askopenfilename, askdirectory
import reduce_conflicts as rc
import pandas as pd
fields = ('Trimester', 'Number of Exam Blocks', 'Number of Reading Periods', 'Path to Classes File', 'Output Directory')

def design_schedule(entries):
    
    trimester = entries["Trimester"].get()
    infile = entries["Path to Classes File"].get()
    outfile = entries["Output Directory"].get()
    num_exam_blocks = int(entries["Number of Exam Blocks"].get())
    num_read_blocks = int(entries["Number of Reading Periods"].get())
    
    data = pd.read_csv(infile)
    print(data.head())

    # get minimum conflict schedule

    # write to file
    
    
def makeform(root, fields):
    entries = {}
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=22, text=field+": ", anchor='w')
        
        if field == "Path to Classes File":
            b = Button(row, text='Choose File',
                    command=(lambda e=entries: set_classes_file_name(e)))
            b.pack(side=RIGHT, padx=5, pady=5)
        
        if field == "Output Directory":
            b = Button(row, text='Choose Directory',
                    command=(lambda e=entries: set_output_directory(e)))
            b.pack(side=RIGHT, padx=5, pady=5)
        
        ent = Entry(row)
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries[field] = ent
        
    return entries

def set_classes_file_name(entries):
    fileName = askopenfilename()
    ent = entries["Path to Classes File"]
    ent.delete(0, len(ent.get()))
    ent.insert(0, str(fileName))

def set_output_directory(entries):
    directory = askdirectory()
    ent = entries["Output Directory"]
    ent.delete(0, len(ent.get()))
    ent.insert(0, str(directory))

if __name__ == '__main__':
    root = Tk()
    # get screen width and height
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    # set window width and height
    root.geometry("%dx%d+0+0" %(w, h))

    ents = makeform(root, fields)
    
    b1 = Button(root, text='Design Exam Schedule',
        command=(lambda e=ents: design_schedule(e)))
    b1.pack(side=LEFT, padx=5, pady=5)

    # set window title
    root.winfo_toplevel().title("Schedule Optimizer")
    
    root.mainloop()
