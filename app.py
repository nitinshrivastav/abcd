#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('heart_failure_clinical_records_dataset.csv')


# In[3]:


x=data.iloc[:,0:-1]
y=data.iloc[:,-1]


# In[4]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.20)


# In[6]:


from sklearn.linear_model import LogisticRegression                               
model=LogisticRegression()


# In[7]:


model.fit(xtrain,ytrain)


# In[ ]:


from flask import Flask,render_template,request
app=Flask(__name__)
@app.route("/")
def abc():
    return render_template('heart.html')
@app.route('/heartfailure', methods=['GET','POST'])
def xyz():
    if(request.method=='POST'):
        age=int(request.form['a1'])
        anemia=int(request.form['an1'])
        crphos=int(request.form['Cp'])
        dia=int(request.form['d1'])
        ej_fra=int(request.form['Ef'])
        HBP=int(request.form['hbp'])
        plate=int(request.form['p1'])
        se_cre=float(request.form['sc1'])
        se_sodi=int(request.form['ss1'])
        gender=int(request.form['x1'])
        smoke=int(request.form['smk1'])
        time=int(request.form['t1'])
        result=model.predict([[age,anemia,crphos,dia,ej_fra,HBP,plate,se_cre,se_sodi,gender,smoke,time]])
        return render_template("heart.html",answer=result)
if __name__=="__main__":
    app.run() 


# In[ ]:





# In[ ]:





# In[ ]:




