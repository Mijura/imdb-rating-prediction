import tkinter as tk
from tkinter import *
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from pandas import *
import numpy as np

class FullScreenApp(object):

	def __init__(self, master, **kwargs):
		self.master=master
		self.master.title('IMDB Rating Data Analysis')
		self.master.wm_state('zoomed')
		self.columns={	1:'color',
						2:'director_name',
						3:'num_critic_for_reviews',
						4:'duration',
						5:'director_facebook_likes',
						6:'actor_3_facebook_likes',
						7:'actor_2_name',
						8:'actor_1_facebook_likes',
						9:'gross',
						10:'genres',
						11:'actor_1_name',
						12:'movie_title',
						13:'num_voted_users',
						14:'cast_total_facebook_likes',
						15:'actor_3_name',
						16:'facenumber_in_poster',
						17:'plot_keywords',
						18:'movie_imdb_link',
						19:'num_user_for_reviews',
						20:'language',
						21:'country',
						22:'content_rating',
						23:'budget',
						24:'title_year',
						25:'actor_2_facebook_likes',
						26:'imdb_score',
						27:'aspect_ratio',
						28:'movie_facebook_likes'}
		
		self.data = pd.read_csv('movie_metadata.csv', encoding = "ISO-8859-1")
		self.set_image()
		self.add_boxplot_frame()
		
	def add_boxplot_frame(self):
		self.boxplot_frame = Frame(self.master)
		self.boxplot_frame.grid(row=0, column=2, padx=10, pady=30, sticky="NW")

		self.var = IntVar()
		
		self.left = Label(self.boxplot_frame, text="Choose column for boxplot:")
		self.left.pack(pady=10)
		
		R1 = Radiobutton(self.boxplot_frame, text="Color", variable=self.var, value=1)
		R1.pack(anchor = W)
		
		R2 = Radiobutton(self.boxplot_frame, text="Country", variable=self.var, value=21)
		R2.pack(anchor = W)

		R3 = Radiobutton(self.boxplot_frame, text="Language", variable=self.var, value=20)
		R3.pack(anchor = W)

		R4 = Radiobutton(self.boxplot_frame, text="Title Year", variable=self.var, value=24)
		R4.pack(anchor = W)

		R5 = Radiobutton(self.boxplot_frame, text="Duration", variable=self.var, value=4)
		R5.pack(anchor = W)
		
		R6 = Radiobutton(self.boxplot_frame, text="Face Number in Poster", variable=self.var, value=16)
		R6.pack(anchor = W)
		
		R7 = Radiobutton(self.boxplot_frame, text="Content Rating", variable=self.var, value=22)
		R7.pack(anchor = W)
		
		R8 = Radiobutton(self.boxplot_frame, text="Aspect Ratio", variable=self.var, value=27)
		R8.pack(anchor = W)
		
		B = Button(self.boxplot_frame, text ="Generate Boxplot", command = self.generate_boxplot)
		B.pack(anchor = W, padx=10, pady=10)
	
	def generate_boxplot(self):
		selection = "You selected the option " + str(self.var.get())
		self.boxplot(self.columns[self.var.get()])
	
	def set_image(self):
		self.image = tk.PhotoImage(file="image.gif")
		self.image = self.image.subsample(2,2)
		self.label = tk.Label(image=self.image)
		self.label.grid(row=0, column=1, padx=10, pady=10)
		#self.label.pack()

	def newselection(self, event):
		self.value_of_combo = self.box.get()
		print(self.value_of_combo)
		
	def modify_name(self, column):
		column = column.replace('_', ' ')
		return column.title()
		
	def boxplot(self, column):
		grp = self.data["imdb_score"].groupby(self.data[column])
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
		plt.title("IMDB Score vs "+self.modify_name(str(column)))
		
		mng = plt.get_current_fig_manager()
		mng.window.showMaximized()
		plt.show()

root = tk.Tk('ads')
app=FullScreenApp(root)
			
root.mainloop()

