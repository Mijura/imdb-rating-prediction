import tkinter as tk
from tkinter import *
import pandas as pd
import matplotlib.pyplot as plt
from pandas import *
import numpy as np
"""
class FullScreenApp(object):
    def __init__(self, master, **kwargs):
        self.master=master
        pad=3
        self._geom='200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth()-pad, master.winfo_screenheight()-pad))
        master.bind('<Escape>',self.toggle_geom)            
    def toggle_geom(self,event):
        geom=self.master.winfo_geometry()
        print(geom,self._geom)
        self.master.geometry(self._geom)
        self._geom=geom

root = tk.Tk()
app=FullScreenApp(root)
image = tk.PhotoImage(file="image.gif")
image = image.subsample(2,2)
label = tk.Label(image=image)
label.pack()
root.mainloop()
"""
d = pd.read_csv('movie_metadata.csv', encoding = "ISO-8859-1")
df = DataFrame(d)
grp = d["imdb_score"].groupby(d['country'])

data=[]
numbers=[]
labels=[]
i=1
for x in grp.groups.keys():
	numbers.append(i)
	labels.append(x)
	d = []
	d.append(list(grp.get_group(x)))
	data.append(d)
	i=i+1
	
plt.boxplot(data)
plt.xticks(numbers, labels, rotation='vertical')
plt.title("IMDB score vs Country")

plt.show()
