#!/usr/bin/env python
# coding: utf-8

# In[1]:


#All necessary libraries imported are shown here

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



# In[2]:


#Importing the Maths Students Dataset and viewing the data
maths_stud = pd.read_csv('maths_students.csv')
math_stud = pd.read_csv('maths_students.csv')
print(maths_stud)


# In[3]:


#Importing Portuguese Students Dataset and viewing the data
por_stud = pd.read_csv('por_students.csv')
port_stud = pd.read_csv('por_students.csv')
print(por_stud)


# In[4]:


#Since there are no names or roll numbers to identify the student, for the sake of developing the prototype, we can include a column with roll numbers


# In[5]:


#Adding the column "roll_numbers" to the maths student dataset

roll_numbers = pd.Series(range(1, len(maths_stud) + 1)).apply(lambda x: 'mat{:03d}'.format(x))
maths_stud['roll_number'] = roll_numbers
#print(math_stud)


# In[6]:


roll_number = pd.Series(range(1, len(math_stud) + 1)).apply(lambda x: 'mat{:03d}'.format(x))
math_stud['roll_number'] = roll_number
#print(math_stud)


# In[7]:


#Adding the column "roll_numbers" to the Portuguese student dataset

roll_numbers1 = pd.Series(range(1, len(por_stud) + 1)).apply(lambda x: 'por{:03d}'.format(x))
por_stud['roll_number'] = roll_numbers1

#print(por_stud)


# In[8]:


roll_numbers11 = pd.Series(range(1, len(port_stud) + 1)).apply(lambda x: 'por{:03d}'.format(x))
port_stud['roll_number'] = roll_numbers11


# In[9]:


#We can add a new column as "Final Mark" which shows the average of G1, G2 and G3 for Maths and POrtuguese


# In[10]:


#Adding a column to see the average grade for Maths G1,G2 and G3 grades

math_stud['Final Mark'] = (math_stud['G1']+math_stud['G2']+math_stud['G3'])/3
maths_stud['Final Mark'] = (maths_stud['G1']+maths_stud['G2']+maths_stud['G3'])/3

#print(maths_stud.head())


# In[11]:


#Adding a column to see the average grade for Portuguese G1,G2 and G3 grades

port_stud['Final Mark'] = (port_stud['G1']+port_stud['G2']+port_stud['G3'])/3
por_stud['Final Mark'] = (por_stud['G1']+por_stud['G2']+por_stud['G3'])/3

#print(port_stud.head())


# In[12]:


#Finding how many students failed or passed in the final test. 
#The test is considered failed if any students has less than 50% in the final score (G1+G2+G3). i.e, < 10

def results(row):
    if row['Final Mark'] < 10:
        return 'Fail'
    else:
        return 'Pass'


# In[13]:


#Adding a column with the results

math_stud['RESULTS'] = math_stud.apply(results, axis = 1)
maths_stud['RESULTS'] = math_stud.apply(results, axis = 1)


# In[14]:


port_stud['RESULTS'] = port_stud.apply(results, axis = 1)
por_stud['RESULTS'] = port_stud.apply(results, axis = 1)


# In[15]:


#Now we have added the Results column to the dataset which shows if the student has Passed or Failed based on the average (Final Mark)
#Any student who got less that 50% of the Final Mark is considered as failed


# In[16]:


#Number of graphs can be created for better understanding of the students data

#The following graph will show Average Maths and portuguese grades in G1,G2 and G3 for 
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
ax1.bar(['G1','G2','G3'], [maths_stud['G1'].mean(),maths_stud['G2'].mean(), maths_stud['G3'].mean()],color='#005C53', edgecolor='black',linewidth=1.5, width=0.5)
ax1.set_title('Average Maths Grades')
ax1.set_xlabel('Assessments')
ax1.set_ylabel('Score')

ax2.bar(['G1','G2','G3'], [por_stud['G1'].mean(),por_stud['G2'].mean(), por_stud['G3'].mean()],color='#042940', edgecolor='black',linewidth=1.5, width=0.5)
ax2.set_title('Average Portuguese Grades')
ax2.set_xlabel('Assessments')
ax2.set_ylabel('Score')

plt.show()


# In[17]:


sns.pairplot(maths_stud[['G1','G2','G3']], height = 4)


# In[18]:


sns.pairplot(por_stud[['G1','G2','G3']], height = 4)


# In[19]:


#Here we can see how the gender of the students as compared to the average G1, G2 and G3 scores for Maths.

fig, (bx1, bx2, bx3) = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(x="sex", y="G1", data=maths_stud, ax=bx1, hue="sex", palette = {"M":"#4AA5FA","F":"#F081E5"})
bx1.set_xlabel("Gender")
bx1.set_ylabel("G1 Score")
bx1.set_title("Maths:G1 Scores by Gender")

sns.boxplot(x="sex", y="G2", data=maths_stud, ax=bx2,hue="sex", palette = {"M":"#4AA5FA","F":"#F081E5"})
bx2.set_xlabel("Gender")
bx2.set_ylabel("G2 Score")
bx2.set_title("Maths:G2 Scores by Gender")

sns.boxplot(x="sex", y="G3", data=maths_stud, ax=bx3,hue="sex", palette = {"M":"#4AA5FA","F":"#F081E5"})
bx3.set_xlabel("Gender")
bx3.set_ylabel("G3 Score")
bx3.set_title("Maths:G3 Scores by Gender")

plt.show()


# In[20]:


#Here we can see how the gender of the students as compared to the average G1, G2 and G3 scores for Portuguese.

fig, (bx1, bx2, bx3) = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(x="sex", y="G1", data=por_stud, ax=bx1, hue="sex", palette = {"M":"#4AA5FA","F":"#F081E5"})
bx1.set_xlabel("Gender")
bx1.set_ylabel("G1 Score")
bx1.set_title("Portuguese:G1 Scores by Gender")

sns.boxplot(x="sex", y="G2", data=por_stud, ax=bx2,hue="sex", palette = {"M":"#4AA5FA","F":"#F081E5"})
bx2.set_xlabel("Gender")
bx2.set_ylabel("G2 Score")
bx2.set_title("Portuguese:G2 Scores by Gender")

sns.boxplot(x="sex", y="G3", data=por_stud, ax=bx3,hue="sex", palette = {"M":"#4AA5FA","F":"#F081E5"})
bx3.set_xlabel("Gender")
bx3.set_ylabel("G3 Score")
bx3.set_title("Portuguese:G3 Scores by Gender")

plt.show()


# In[21]:


#Using SVM Model to predict the students who passed or failed in Maths


# In[22]:


#Tranforming the dataset for the model
#All categorical columns are transformed to numerical representation

# Creating LabelEncoder for transformation
label_encoder = LabelEncoder()

#Testing with the "school" column

encoded_school = label_encoder.fit_transform(maths_stud["school"])

maths_stud["school"] = encoded_school





# In[23]:


#Doing the same transformation for rest of the selected categorical columns

#list of selected columns

categorical_columns = ["sex","RESULTS" ,"address", "Pstatus", "guardian", "activities", "nursery", "internet", "romantic"]

# Iterating over the categorical columns to encode them

for column in categorical_columns:
    encoded_column = label_encoder.fit_transform(maths_stud[column])
    maths_stud[column] = encoded_column


# In[24]:


#Creating New dataset for the model
#This model contains selected columns 

selected_columns = ["sex", "address", "Pstatus", "guardian", "activities", "nursery", "internet", "romantic","G1","G2", "G3","RESULTS", "Final Mark", "absences", "health", "Walc", "Dalc", "goout", "freetime", "famrel", "studytime", "traveltime", "Medu", "Fedu", "age"  ]
maths_stud2 = maths_stud.loc[:, selected_columns].copy()

#Check the new dataset to see if all columns are numerical
#print(maths_stud2)


# In[25]:


#SVM MODEL#


# In[26]:


X = maths_stud2.drop(['RESULTS'], axis=1)
y = maths_stud2['RESULTS']

#Splitting the dataset for training and testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training the SVM model using Support Vector Classification (SVC)

svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(X_train, y_train)



# In[27]:


#Model for prediction 
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))


# In[28]:


#Checking the accuracy using confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[29]:


sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[30]:


print(maths_stud['roll_number'])


# In[31]:


student_ID = maths_stud['roll_number']


# In[32]:


maths_pred = maths_stud2.drop('RESULTS',axis=1)

mathsprediction = svm.predict(maths_pred)


# In[33]:


maths_results = pd.DataFrame({'Student_ID': student_ID, 'Results': mathsprediction})

#To show the results in categorical value
maths_results['Results'] = maths_results['Results'].map({0:'Failed',1:'Passed'})

print(maths_results)


# In[34]:


result_counts = maths_results['Results'].value_counts()
print(result_counts)


# In[35]:


#This shows, 232 students are predicted to be passed and 163 students are predicted to be failed in Maths


# In[36]:


##### Doing the same process for Portuguese Students######


# In[37]:


#Tranforming the dataset for the model
#All categorical columns are transformed to numerical representation


#Testing with the "school" column

encoded_school2 = label_encoder.fit_transform(por_stud["school"])

por_stud["school"] = encoded_school2


#Doing the same transformation for rest of the selected categorical columns

#list of selected columns

categorical_columns2 = ["sex","RESULTS" ,"address", "Pstatus", "guardian", "activities", "nursery", "internet", "romantic"]

# Iterating over the categorical columns to encode them

for column in categorical_columns2:
    encoded_column = label_encoder.fit_transform(por_stud[column])
    por_stud[column] = encoded_column


# In[38]:


#Creating New dataset for the model with selected columns

selected_columns2 = ["sex", "address", "Pstatus", "guardian", "activities", "nursery", "internet", "romantic","G1","G2", "G3","RESULTS", "Final Mark", "absences", "health", "Walc", "Dalc", "goout", "freetime", "famrel", "studytime", "traveltime", "Medu", "Fedu", "age"  ]
por_stud2 = por_stud.loc[:, selected_columns2].copy()

#Check the new dataset to see if all columns are numerical
#print(maths_stud2)


# In[39]:


##SVM MODEL#

X = por_stud2.drop(['RESULTS'], axis=1)
y = por_stud2['RESULTS']

#Splitting the dataset for training and testing

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)

#Training the SVM model using Support Vector Classification (SVC)

svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(X_train2, y_train2)



# In[40]:


#Model for prediction 
y_pred2 = svm.predict(X_test2)
print(classification_report(y_test2, y_pred2))


# In[41]:


#Checking the accuracy using confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm2 = confusion_matrix(y_test2, y_pred2)
accuracy2 = accuracy_score(y_test2, y_pred2)
print("Accuracy:", accuracy)

sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[42]:


#Predicting the results of the entire students

porstudent_ID = por_stud['roll_number']
por_pred = por_stud2.drop('RESULTS',axis=1)

porprediction = svm.predict(por_pred)
por_results = pd.DataFrame({'Student_ID': porstudent_ID, 'Results': porprediction})

#To show the results in categorical value
por_results['Results'] = por_results['Results'].map({0:'Failed',1:'Passed'})

#print(por_results)


# In[43]:


porresult_counts = por_results['Results'].value_counts()
print(porresult_counts)


# In[44]:


#This shows 492 students has passed while 157 has failed in Portuguese


# In[45]:


#Creating different visualization which can be added in the web application


# In[46]:


# Get the counts for Maths results
result_counts_maths = maths_results['Results'].value_counts()

# Get the counts for Portuguese results
result_counts_por = por_results['Results'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))

# Plot for Maths results prediction
plt.subplot(121)
result_counts_maths.plot(kind='bar', color=['red', 'green'])
plt.title('Maths Results Prediction')
plt.xlabel('Results')
plt.ylabel('Number of Students')

#Plot for Portuguese Result Prediction
plt.subplot(122)
result_counts_por.plot(kind='bar', color=['red', 'green'])
plt.title('Portuguese Results Prediction')
plt.xlabel('Results')
plt.ylabel('Number of Students')

plt.tight_layout()
plt.show()


# In[47]:


#"ipywidgets" can be used to combined multiple graphs and make the visualization interactive
# Creating a dropdown widget which shows the average Maths and Portuguese grades

options = ['Maths', 'Portuguese']
dropdown = widgets.Dropdown(options=options, value=options[0], description='Select subject:')

# Define a function to display the selected graph
def display_averagemarks(selected_option):
    plt.clf()
    fig, ax = plt.subplots(figsize=(7,5))
    if selected_option == 'Maths':
        ax.bar(['G1','G2','G3'], [math_stud['G1'].mean(),math_stud['G2'].mean(), math_stud['G3'].mean()],color='#005C53', edgecolor='black',linewidth=1.5, width=0.5)
        ax.set_title('Average Maths Grades')
        ax.set_xlabel('Assessments')
        ax.set_ylabel('Score')
        plt.show()
    else:
        ax.bar(['G1','G2','G3'], [port_stud['G1'].mean(),port_stud['G2'].mean(), port_stud['G3'].mean()],color='#042940', edgecolor='black',linewidth=1.5, width=0.5)
        ax.set_title('Average Portuguese Grades')
        ax.set_xlabel('Assessments')
        ax.set_ylabel('Score')
        plt.show()
        
# Link the dropdown widget with the function that displays the selected graph
widgets.interact(display_averagemarks, selected_option=dropdown)


# In[48]:


result_count = math_stud['RESULTS'].value_counts()
plt.bar(result_count.index, result_count.values,color=['#92FA91', '#F03F35'])

for i, v in enumerate(result_count.values):
    plt.text(i, v+1, str(v), color='black', ha='center')

plt.xlabel('Pass / Fail')
plt.ylabel('Number of students')
plt.title('Number of students who failed in the Final score for Maths')

plt.show()


# In[49]:


#We can see that 231 students passed in the aggregated score for Maths while 164 students failed.


# In[50]:


result_counts = port_stud['RESULTS'].value_counts()
plt.bar(result_counts.index, result_counts.values,color=['#92FA91', '#F03F35'])

for i, v in enumerate(result_counts.values):
    plt.text(i, v+1, str(v), color='black', ha='center')

plt.xlabel('Pass / Fail')
plt.ylabel('Number of students')
plt.title('Number of students who failed in the Final score for Portuguese')

plt.show()


# In[51]:


#We can see that 492 students passed in the Portuguese aggregated score while 157 students failed


# In[52]:


#We can use ipywidgets again to combine these graphs


# In[53]:


options = ['Maths', 'Portuguese']
dropdown = widgets.Dropdown(options=options, value=options[0], description='Select subject:')

def display_passfail(selected_option):
    fig, ax = plt.subplots(figsize=(10,5))
    if selected_option == 'Maths':
        result_count = math_stud['RESULTS'].value_counts()
        ax.bar(result_count.index, result_count.values,color=['#92FA91', '#F03F35'])
        for i, v in enumerate(result_count.values):
            ax.text(i, v+1, str(v), color='black', ha='center')
        ax.set_xlabel('Pass / Fail')
        ax.set_ylabel('Number of students')
        ax.set_title('Number of students who failed in the Final score for Maths')
        plt.show()
    else:
        result_counts = port_stud['RESULTS'].value_counts()
        ax.bar(result_counts.index, result_counts.values,color=['#92FA91', '#F03F35'])
        for i, v in enumerate(result_counts.values):
            ax.text(i, v+1, str(v), color='black', ha='center')
        ax.set_xlabel('Pass / Fail')
        ax.set_ylabel('Number of students')
        ax.set_title('Number of students who failed in the Final score for Portuguese')
        plt.show()
        
# Link the dropdown widget with the function that displays the selected graph
widgets.interact(display_passfail, selected_option=dropdown)


# In[54]:


#Comparing the performance of students using Internet Access
#Creating a drop-down widget for both the subjects

options = ['Maths', 'Portuguese']
dropdown = widgets.Dropdown(options=options, value=options[0], description='Select subject:')

def display_internet(selected_option):
    if selected_option == 'Maths':
        # Generate the crosstab and plot the graph for Maths
        internet_results = pd.crosstab(math_stud['internet'], math_stud['RESULTS'])
        ax = internet_results.plot(kind='bar', color=['#F03F35','#92FA91'])
        ax.set_xlabel('Internet Access')
        ax.set_ylabel('Number of Students')
        ax.set_title('Maths:Passed and Failed Students by Internet Access')
        plt.show()
    else:
        # Generate the crosstab and plot the graph for Portuguese
        internet_results1 = pd.crosstab(port_stud['internet'], port_stud['RESULTS'])
        ax = internet_results1.plot(kind='bar', color=['#F03F35','#92FA91'])
        ax.set_xlabel('Internet Access')
        ax.set_ylabel('Number of Students')
        ax.set_title('Por:Passed and Failed Students by Internet Access')
        plt.show()

# Link the dropdown widget with the function that displays the selected graph
widgets.interact(display_internet, selected_option=dropdown)


# In[55]:


#Visualization of different correlation of factors with grades

#Scatter plot to show the correlation of Study time with grades

plt.scatter(math_stud['studytime'],math_stud['Final Mark'], c = math_stud['Final Mark'].apply(lambda x: '#F03F35' if x < 10 else '#92FA91'))
plt.xlabel('Study Time')
plt.ylabel('Final Grade')
plt.title('Affects of study time on the Final Grade of Maths')
plt.show()

#Calculating the correlation coefficient

r = math_stud['studytime'].corr(math_stud['Final Mark'])

print('Correlation between Final Grade in Maths and Study Time', r)

#Green shows the passing grades and red shows the failures ( 10 and above is passing grade)


# In[56]:


#We can see that there is a positive correlation. However, it is weak correlation. The students who study longer is shown to be performing only slightly better. 


# In[57]:


#Scatter plot to show the correlation of Study time of Portuguese with grades

plt.scatter(port_stud['studytime'],port_stud['Final Mark'], c = port_stud['Final Mark'].apply(lambda x: '#F03F35' if x < 10 else '#92FA91'))
plt.xlabel('Study Time')
plt.ylabel('Final Grade')
plt.title('Affects of study time on the Final Grade of Portuguese')
plt.show()

#Calculating the correlation coefficient

r = port_stud['studytime'].corr(port_stud['Final Mark'])

print('Correlation between Final Grade in Portuguese and Study Time', r)

#Green shows the passing grades and red shows the failures ( 10 and above is passing grade)


# In[58]:


#There is a moderately slightly better positive correlation between Portuguese final grade and study time. However,the relation is still not significant. 


# In[59]:


options = ['Maths', 'Portuguese']
dropdown = widgets.Dropdown(options=options, value=options[0], description='Select subject')

def display_studytime(selected_option):
    if selected_option == 'Maths':
        fig, ax = plt.subplots(figsize=(10,5))
        plt.scatter(math_stud['studytime'], math_stud['Final Mark'], c=math_stud['Final Mark'].apply(lambda x: '#F03F35' if x < 10 else '#92FA91'))
        r = math_stud['studytime'].corr(math_stud['Final Mark'])
        plt.xlabel('Study Time')
        plt.ylabel('Final Grade')
        plt.title(f'Effect of study time on the Final Grade of Maths\nCorrelation: {r:.2f}')
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(10,5))
        plt.scatter(por_stud['studytime'], port_stud['Final Mark'], c=port_stud['Final Mark'].apply(lambda x: '#F03F35' if x < 10 else '#92FA91'))
        r = port_stud['studytime'].corr(port_stud['Final Mark'])
        plt.xlabel('Study Time')
        plt.ylabel('Final Grade')
        plt.title(f'Effect of study time on the Final Grade of Portuguese\nCorrelation: {r:.2f}')
        plt.show()

widgets.interact(display_studytime, selected_option=dropdown)


# In[60]:


#Visualising how the students are divided based on the guardians
#In the dataset, there are Father, Mother and Others denoted as guardians


# In[61]:


import matplotlib.pyplot as plt

guardian_counts = math_stud['guardian'].value_counts()

fig, ax = plt.subplots(figsize=(5, 5))
guardian_counts.plot(kind='bar', color=['#F65DB1', '#C265EB','#755DF6'])
ax.set_xticklabels(['Mother', 'Father', 'Other'], rotation=0)
ax.set_xlabel('Guardian')
ax.set_ylabel('Number of Students')
ax.set_title('Counts of Students with thier guardians')
plt.show()


# In[62]:


#We can see that there are more students with Mother as their guardians than Father


# In[63]:


#Visualizing how the Results of the students are displayed based on their guardians


# In[64]:


print(maths_stud['guardian'])


# In[65]:


options = ['mother', 'father', 'other']
dropdown = widgets.Dropdown(options=options, value=options[0], description='Guardian:')

def display_pass_fail(guardian):
    if guardian == 'mother':
        guardian_stud = math_stud[math_stud['guardian'] == 'mother']
        title = 'Maths Subject Pass/Fail Counts for Students with Mother as Guardian'
    elif guardian == 'father':
        guardian_stud = math_stud[math_stud['guardian'] == 'father']
        title = 'Maths Subject Pass/Fail Counts for Students with Father as Guardian'
    else:
        guardian_stud = math_stud[~math_stud['guardian'].isin(['mother', 'father'])]
        title = 'Maths Subject Pass/Fail Counts for Students with Other Guardian'

    guardian_results = guardian_stud.groupby('RESULTS')['RESULTS'].count()

    fig, ax = plt.subplots(figsize=(10, 5))
    guardian_results.plot(kind='bar', color=['#F03F35', '#92FA91'])
    ax.set_xticklabels(['Fail', 'Pass'], rotation=0)
    ax.set_xlabel('Pass/Fail Status')
    ax.set_ylabel('Number of Students')
    ax.set_title(title)
    plt.show()

widgets.interact(display_pass_fail, guardian=dropdown)


# In[66]:


#################### Creating the website application using DASH###############


# In[ ]:


#Defining the App

app = dash.Dash(__name__)

#I want to show the average Maths and Portuguese grades on the top of the page
#Therefore it is defined above the dash layout

#here plotly.py is used to define the figure

fig = go.Figure()

fig.add_trace(go.Bar(
    x=['G1', 'G2', 'G3'],
    y=[math_stud['G1'].mean(), math_stud['G2'].mean(), math_stud['G3'].mean()],
    name='Average Maths Grades',
    marker=dict(color='#EB4B98'),
    width=0.5,
    marker_line=dict(color='black', width=1.5),
))

fig.add_trace(go.Bar(
    x=['G1', 'G2', 'G3'],
    y=[port_stud['G1'].mean(), port_stud['G2'].mean(), port_stud['G3'].mean()],
    name='Average Portuguese Grades',
    marker=dict(color='#54E8BD'),
    width=0.5,
    marker_line=dict(color='black', width=1.5),
))

fig.update_layout(
    title={'text':'Average Grades','x':0.5},
    xaxis_title='Assessments',
    yaxis_title='Score',
    width=800,
    height=500,
    margin=dict(l=40, r=40, t=80, b=40)
)

#Now App Layout of the Dash app is defined here
#html.Div can be used to hold all the components togethers as needed. 


app.layout = html.Div([
    html.H1("Students Performance Analysis Dashboard", style={'text-align':'center','font-family': 'Montserrat, sans-serif','font-weight':'bold','font-size':'60px','color':'#6a00ff'}), #this shows the header title
    dcc.Graph(figure=fig),
    dcc.Dropdown(id="Select_Subject",
                options=[{"label":"Maths", "value":"Maths"},
                        {"label":"Portuguese", "value":"Portuguese"}],
                multi=False,
                value="Maths",
                style={'width':"40%"}
                ),
    html.Div(
        [
            dcc.Graph(id='final-grade-vs-studytime-graph'),
            dcc.Graph(id='internet_access_graph')
        ]
    ),
#     dcc.Dropdown(id="Select_Gender",
#                 options=[{"label":"Male", "value":"Male"},
#                         {"label":"Female", "value":"Female"}],
#                 multi=False,
#                 value="Male",
#                 style={'width':"40%"}
#                 ),

#For Portuguese Prediction layout
    
    html.Div([
        html.H1("Predicted result for Portuguese Students", style={'text-align':'center','font-family':'Arial,sans-serif','font-weight':'bold','color':'#DB0252'}), 
        dcc.Dropdown(
        id='Student_ID',
        options=[{'label':str(Student_ID),'value': Student_ID} for Student_ID in por_results['Student_ID']],
        placeholder = "Select a Student"
        ),
        html.Div(id='result-output')
    ]),
    
#For Maths Prediction Layout
    
        html.Div([
        html.H1("Predicted result for Maths Students", style={'text-align':'center','font-family':'Arial,sans-serif','font-weight':'bold','color':'#DB0252'}), 
        dcc.Dropdown(
        id='MStudent_ID',
        options=[{'label':str(Student_ID),'value': Student_ID} for Student_ID in maths_results['Student_ID']],
        placeholder = "Select a Student"
        ),
        html.Div(id='maths_result_output')
    ]),
    
],
) 



@app.callback(Output('final-grade-vs-studytime-graph', 'figure'),
              [Input('Select_Subject', 'value')])
def update_final_grade_vs_studytime_graph(selected_subject):
    if selected_subject == 'Maths':
        data = math_stud
        title = 'Effect of study time on the Final Grade of Maths'
    else:
        data = port_stud
        title = 'Effect of study time on the Final Grade of Portuguese'
    r = data['studytime'].corr(data['Final Mark'])

    # Create the scatter plot with color scale
    fig = px.scatter(data, x='studytime', y='Final Mark', color='Final Mark',
                     color_continuous_scale=px.colors.sequential.RdBu,
                     range_color=[0, 20])
    fig.update_layout(title=title, xaxis_title='Study Time', yaxis_title='Final Grade',
                      coloraxis=dict(colorbar_title='Final Grade'))
    fig.add_annotation(text=f'Correlation: {r:.2f}', xref='paper', yref='paper',
                       x=1, y=1, showarrow=False, font=dict(size=16))
    return fig

@app.callback(Output('internet_access_graph', 'figure'),
              [Input('Select_Subject', 'value')])
def internet_access(selected_subject):
    # Select the appropriate data and create histogram
    if selected_subject == 'Maths':
        data = math_stud
        title = 'Maths:Passed and Failed Students by Internet Access'
    else:
        data = port_stud
        title = 'Portuguese:Passed and Failed Students by Internet Access'

    fig = px.bar(data, x='internet', color='RESULTS', barmode='group',color_discrete_map={'Fail': '#F03F35', 'Pass': '#56FC0D'})
    fig.update_layout(title=title, xaxis_title='Internet Access', yaxis_title='Count')


    return fig

#For Portuguese Prediction Dashboard
@app.callback(
Output('result-output','children'),
[Input('Student_ID','value')]
)

def update_result(porstudent_ID):
    if porstudent_ID is None:
        return ''
    result = por_results.loc[por_results['Student_ID'] == porstudent_ID, 'Results'].iloc[0]
    if result == 'Passed':
        return html.H3(f'Student {porstudent_ID} is more likely to PASS.')
    elif result == 'Failed':
        return html.H3(f'Student {porstudent_ID} is more likely to FAIL. Give additional support')
    else:
        return html.H3('No result available for the selected student.')

#for Maths Prediction dashboard

@app.callback(
Output('maths_result_output','children'),
[Input('MStudent_ID','value')]
)

def update_math_result(student_ID):
    if student_ID is None:
        return ''
    results = maths_results.loc[maths_results['Student_ID'] == student_ID, 'Results'].iloc[0]
    if results == 'Passed':
        return html.H3(f'Student {student_ID} is more likely to PASS.')
    elif results == 'Failed':
        return html.H3(f'Student {student_ID} is more likely to FAIL.Give additional Support')
    else:
        return html.H3('No result available for the selected student.')
  
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




