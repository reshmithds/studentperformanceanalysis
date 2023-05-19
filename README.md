##Students Performance Analysis and Prediction

This web application serves as a prototype designed specifically for educational institutions seeking to monitor their students' performance. It offers an interactive dashboard that presents students' performance through interactive visualizations, allowing educators to easily interpret the data.

Moreover, the web application includes a feature that enables users to select individual students and access the predictions of Maths and Portuguese results. This functionality empowers educators to identify students at risk of failing and provide them with the necessary additional support.


##INSTALLATION

This web application is a prototype created using Dash framework of Python. Therefore, a python tool like Jupyter Notebook must be installed and necessary libraries imported to run the application. Jupyter Notebook is the suggested tool as it is used to create the application. However, any tool which operates python can be used.

##All the files in the zip files should be extracted for using the Desktop App 

#Steps to follow
EXTRACT THE FILE FROM ZIP

1. Install Python
2. install Jupyter Notebook

3. Libraries to import and use in Jupyter Notebook. Used for the visualization, interative features and dashboard. 

import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from flask import Flask, render_template
import plotly.graph_objs as go
import plotly.express as px
import ipywidgets as widgets
import dash
from dash import Dash, html, dcc, Input, Output
from ipywidgets import SelectMultiple

4. Copy the studentperformance.py code from the file and run it. 
5. It is suggested to Restart the kernel and re-run the entire secript everytime. 
6. Once the the entire script is run, you can find a link in the last output. Click that link to open the app on your browser
7. For Desktop app, you can open the extracted file "Desktop App - nodejs" and find the app inside - "Student Performance Analysis App"
8. Double click and open the app. 


##How to use the Student Performance Analysis App

1. Use the scrollbar to go up and down in the web app
2. Use the drop-down option where provided to select the required option
3. The visualization is changed by the option selected
4. Use the drop-down to select the student ID, this will give the prediction on whether the student is more likely to pass or fail. 

NOTE: The desktop app would only work if the code if the Python code is run in the background.  

#CONTRIBUTING

Any students or others who are willing to contribute to this project are welcome. The web application is created using Dash framework and therefore any modification should be done within this framework.

Expected area of contribution are:

1. Interactive Visualizations
2. User Interface
3. Desktop Application

Interested people can connect with me on reshmithramesh[at]gmail[dot]com

##CREDITS

Several journals, web articles are reviewed for this project.
 
For the web application the following project is used as a reference: https://github.com/Coding-with-Adam/Dash-by-Plotly/blob/master/Other/Dash_Introduction/intro.py

For customization in Dash web app: https://dash.plotly.com/layout

For colour palette: https://color.adobe.com/create/color-wheel

For converting the web app to Desktop App: https://github.com/nativefier/nativefier/issues/1473 

##LICENSE

This project is created as a Prototype for the assignment of CETM46 - Data Science Product Development. This product is created without any license and modification can be made without any further permissions. However, kindly note that this is an incomplete prototype and therefore should not be used for any valid student analysis. 



