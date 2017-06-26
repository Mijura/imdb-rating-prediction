import tkinter as tk
from tkinter import *
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from pandas import *
import numpy as np
import seaborn as sns

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
		self.add_scatter_plot_frame()
		self.add_boxplot_frame()
		self.add_correlation_frame()
	
	def add_scatter_plot_frame(self):
		self.basic_plot_frame = Frame(self.master)
		self.basic_plot_frame.grid(row=0, column=2, padx=50, pady=30, sticky="NW")

		self.var1 = IntVar()
		
		self.left = Label(self.basic_plot_frame, text="Choose column for scatter plot:")
		self.left.pack(pady=10)
		
		R1 = Radiobutton(self.basic_plot_frame, text="Number of Reviews by Critics", variable=self.var1, value=3)
		R1.pack(anchor = W)
		
		R2 = Radiobutton(self.basic_plot_frame, text="Duration", variable=self.var1, value=4)
		R2.pack(anchor = W)

		R3 = Radiobutton(self.basic_plot_frame, text="Director Facebook Likes", variable=self.var1, value=5)
		R3.pack(anchor = W)

		R4 = Radiobutton(self.basic_plot_frame, text="Actor 1 Facebook Likes", variable=self.var1, value=8)
		R4.pack(anchor = W)

		R5 = Radiobutton(self.basic_plot_frame, text="Actor 2 Facebook Likes", variable=self.var1, value=25)
		R5.pack(anchor = W)
		
		R6 = Radiobutton(self.basic_plot_frame, text="Actor 3 Facebook Likes", variable=self.var1, value=6)
		R6.pack(anchor = W)
		
		R7 = Radiobutton(self.basic_plot_frame, text="Gross", variable=self.var1, value=9)
		R7.pack(anchor = W)
		
		R8 = Radiobutton(self.basic_plot_frame, text="Number Voted Users", variable=self.var1, value=13)
		R8.pack(anchor = W)
		
		R9 = Radiobutton(self.basic_plot_frame, text="Cast Total Facebook Likes", variable=self.var1, value=14)
		R9.pack(anchor = W)
		
		R10 = Radiobutton(self.basic_plot_frame, text="Budget", variable=self.var1, value=23)
		R10.pack(anchor = W)
		
		R11 = Radiobutton(self.basic_plot_frame, text="Movie Facebook Likes", variable=self.var1, value=28)
		R11.pack(anchor = W)
		
		B = Button(self.basic_plot_frame, text ="Generate Scatter Plot", command = self.generate_scatter_plot)
		B.pack(anchor = W, padx=10, pady=10)
	
	def add_boxplot_frame(self):
		self.boxplot_frame = Frame(self.master)
		self.boxplot_frame.grid(row=0, column=3, padx=50, pady=30, sticky="NW")

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
		
	def add_correlation_frame(self):
		self.correlation_frame = Frame(self.master)
		self.correlation_frame.grid(row=0, column=4, padx=50, pady=30, sticky="NW")
		
		self.left = Label(self.correlation_frame, text="To see correlation press button:")
		self.left.pack(pady=10)
		
		B = Button(self.correlation_frame, text ="Generate Correlation", command = self.generate_correlation)
		B.pack(anchor = W, padx=10, pady=10)
	
	def generate_scatter_plot(self):
		self.basic_plot(self.columns[self.var1.get()])
	
	def generate_boxplot(self):
		self.boxplot(self.columns[self.var.get()])
		
	def generate_correlation(self):
		m=DataFrame(self.data).corr()
		plt.figure(figsize=(10,10))
		sns.heatmap(m,vmax=1,annot=True)
		plt.yticks(rotation=0) 
		plt.xticks(rotation=90)
		mng = plt.get_current_fig_manager()
		mng.window.showMaximized()
		plt.show()
	
	def set_image(self):
		self.image = tk.PhotoImage(file="image.gif")
		self.image = self.image.subsample(2,2)
		self.label = tk.Label(image=self.image)
		self.label.grid(row=0, column=1, padx=10, pady=10)
		
	def modify_name(self, column):
		column = column.replace('_', ' ')
		return column.title()
	
	def basic_plot(self, column):
		grp = self.data["imdb_score"].groupby(self.data[column])
		data=[]
		numbers=[]
		labels=[]
		i=1
		x_axis = []
		y_axis = []
		for x in grp.groups.keys():
			numbers.append(i)
			labels.append(x)
			temp = list(grp.get_group(x))
			x_axis = x_axis + temp
			y_axis = y_axis + [int(x)]*len(temp)
			
			i=i+1
	
		plt.scatter(y_axis, x_axis, facecolors='none', edgecolors='c')
		#plt.xticks(numbers, labels, rotation='vertical')
		plt.title("IMDB Score vs "+self.modify_name(str(column)))
		
		plt.xlabel(self.modify_name(str(column)), fontsize=14)
		plt.ylabel('IMDB Score', fontsize=14)
		
		mng = plt.get_current_fig_manager()
		mng.window.showMaximized()
		
		plt.show()
	
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

