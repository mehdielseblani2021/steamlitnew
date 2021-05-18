import pandas as pd
import numpy as np
import streamlit as st
import plotly as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

DATA = pd.read_csv('Telco_Churn1.csv')
@st.cache

def load_data(nrows):
    data = pd.read_csv(DATA, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data


###Visualizing Data with Plotly
###Churn
Churn = DATA.churn

Churn = Churn.astype(str)
Churn= Churn.apply(lambda x: x.replace('1','Churn'))
Churn= Churn.apply(lambda x: x.replace('0',' No Churn'))
churn_counts = Churn.value_counts()

x = [('Churn',churn_counts[0]), ('No Churn', churn_counts[1])]
Churn1 = pd.DataFrame(x, columns=['Name', 'Count'])
fig = px.pie(Churn1, values='Count', names='Name', color_discrete_sequence=px.colors.sequential.Plotly3)


###Gender
Gender = DATA.gender.value_counts()
gen = [('Male',Gender[0]), ('Female', Gender[1])]
Gender1 = pd.DataFrame(gen,columns = ['Name','Value'])
fig2 = px.pie(Gender1, values='Value', names='Name', hole = 0.3, color_discrete_sequence=px.colors.sequential.Agsunset)
fig2.update_traces(textposition='inside', textinfo='percent+label')
fig2.update_layout(
    title_text='Sex of Clients ', showlegend = False)
##Charges
import numpy as np
x0 = DATA.monthlycharges
x1 = DATA.totalcharges

fig702 = go.Figure()
fig702.add_trace(go.Histogram(
    x=x0,
    histnorm='percent',
    name='control'
    ,
    marker_color='#EB89B5',
    opacity=0.75))
fig702.update_layout(
    title_text='Sampled Results', # title of plot
    xaxis_title_text='Value', # xaxis label
    yaxis_title_text='Count', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1) # gap between bars of the same location coordinates

fig703 = go.Figure()
fig703.add_trace(go.Histogram(
    x=x1,
    histnorm='percent',
    name='control'
    ,
    marker_color='#EB89B5',
    opacity=0.75))
fig703.update_layout(
    title_text='Sampled Results', # title of plot
    xaxis_title_text='Value', # xaxis label
    yaxis_title_text='Count', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1) # gap between bars of the same location coordinates



avg_monthly = DATA.monthlycharges.mean()
sum_monthly = DATA.monthlycharges.sum()
avg_total = DATA.totalcharges.mean()
sum_total = DATA.totalcharges.sum()
charges = [('Average Monthly Charges',avg_monthly), ('Average Total Charges',avg_total)]
Charges = pd.DataFrame(charges,columns = ['Name','Value'])

#st.write('Average Monthly Charges:')
#st.write(avg_monthly)




##Client Information
Tenure = DATA.tenure.value_counts()
avg_tenure = Tenure.mean()
fig205 = avg_tenure

Partner = DATA.partner.value_counts()
partner_service = [('Has a Partner',Partner[0]), ('Single',Partner[1])]
Partner1 = pd.DataFrame(partner_service,columns = ['Name','Value'])

fig206 = px.pie(Partner1, values='Value', names='Name',hole = 0.3, color_discrete_sequence=px.colors.sequential.Agsunset)
fig206.update_traces(textposition='inside', textinfo='percent+label')
fig206.update_layout(
    title_text='Relationship Status of Clients', # title of plot
    xaxis_title_text='', # xaxis label
    yaxis_title_text='', # yaxis label
    showlegend = False,
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1) # gap between bars of the same location coordinates

Dependents = DATA.dependents.value_counts()
dependents_service = [('Has Dependents',Dependents[0]), ('No Dependents',Dependents[1])]
Dependents1 = pd.DataFrame(dependents_service,columns = ['Name','Value'])

fig207 = go.Figure(go.Bar(
            y= Dependents1.Name,
            x= Dependents1.Value,
            marker_color='#E23CBF',
            opacity=0.75,
            orientation='h'))

fig207.update_layout(
    title_text='Number of Clients with Dependents', # title of plot
    xaxis_title_text='', # xaxis label
    yaxis_title_text='', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1) # gap between bars of the same location coordinates

Seniorcitizen = DATA.seniorcitizen.value_counts()
seniorcitizen_service = [('Senior Citizen',Seniorcitizen[0]), ('Not a Senior Citizen',Seniorcitizen[1])]
Seniorcitizen1 = pd.DataFrame(seniorcitizen_service,columns = ['Name','Value'])

fig208 = go.Figure(go.Bar(
            y= Seniorcitizen1.Name,
            x= Seniorcitizen1.Value,
            marker_color='#743296',
            opacity=0.75,
            orientation='h'))

fig208.update_layout(
    title_text='Number of Clients Who are Senior Citizens', # title of plot
    xaxis_title_text='', # xaxis label
    yaxis_title_text='', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1) # gap between bars of the same location coordinates
Internet = DATA.internetservice.value_counts()
internet_service = [('Fiber Optic',Internet[0]), ('DSL',Internet[1]),('None',Internet[2])]
Internet1 = pd.DataFrame(internet_service,columns = ['Name','Value'])

fig3 = go.Figure(go.Bar(
            y= Internet1.Name,
            x= Internet1.Value,
            marker_color='#F15A60',
            opacity=0.75,
            orientation='h'))

fig3.update_layout(
title_text='Number of Clients Subscribed to the Internet', # title of plot
xaxis_title_text='', # xaxis label
yaxis_title_text='', # yaxis label
bargap=0.2, # gap between bars of adjacent location coordinates
bargroupgap=0.1) # gap between bars of the same location coordinates

Phone = DATA.phoneservice.value_counts()
phone_service = [('Have Phone Service',Phone[0]), ('No Phone Service',Phone[1])]
Phone1 = pd.DataFrame(phone_service,columns = ['Name','Value'])

fig31 = px.pie(Phone1, values='Value', names='Name', hole = .3, color_discrete_sequence=px.colors.diverging.PiYG)
fig31.update_traces(textposition='inside', textinfo='percent+label')

fig31.update_layout(
    title_text='Number of Clients Subscribed to Phone Service', # title of plot
    xaxis_title_text='Status', # xaxis label
    yaxis_title_text='', # yaxis label
    showlegend=False,
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1) # gap between bars of the same location coordinates

MultipleLines = DATA.multiplelines.value_counts()
MultipleLines_service = [('2+ Lines',MultipleLines[0]), ('1 Line',MultipleLines[1]),('No Phone',MultipleLines[2])]
MultipleLines1 = pd.DataFrame(MultipleLines_service,columns = ['Name','Value'])

fig32 = px.pie(MultipleLines1, values='Value', names='Name',hole = .3, color_discrete_sequence=px.colors.diverging.PiYG)
fig32.update_traces(textposition='inside', textinfo='percent+label')
fig32.update_layout(
    title_text='Number of Clients Subscribed to Multiple Phone Lines', # title of plot
    xaxis_title_text='Status', # xaxis label
    yaxis_title_text='', # yaxis label
    showlegend = False,
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1) # gap between bars of the same location coordinates

Onlinesecurity = DATA.onlinesecurity.value_counts()
onlinesecurity_service = [('Has Online Security',Onlinesecurity[0]), ('No Online Security',Onlinesecurity[1]),('No Internet',Onlinesecurity[2])]
Onlinesecurity1 = pd.DataFrame(onlinesecurity_service,columns = ['Name','Value'])

fig33 = px.pie(Onlinesecurity1, values='Value', names='Name',hole = 0.3, color_discrete_sequence=px.colors.sequential.Agsunset)
fig33.update_traces(textposition='inside', textinfo='percent+label')
fig33.update_layout(
    title_text='Number of Clients Subscribed to Online Security', # title of plot
    xaxis_title_text='Status', # xaxis label
    yaxis_title_text='', # yaxis label
    showlegend= False,
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1) # gap between bars of the same location coordinates
Onlinebackup = DATA.onlinebackup.value_counts()
onlinebackup_service = [('Has Online Backup',Onlinebackup[0]), ('No Online Backup',Onlinebackup[1]),('No Internet',Onlinebackup[2])]
Onlinebackup1 = pd.DataFrame(onlinebackup_service,columns = ['Name','Value'])

fig34 = px.pie(Onlinebackup1, values='Value', names='Name', hole = 0.3, color_discrete_sequence=px.colors.sequential.Agsunset)
fig34.update_traces(textposition='inside', textinfo='percent+label')
fig34.update_layout(
    title_text='Number of Clients Subscribed to Online Backup', # title of plot
    xaxis_title_text='', # xaxis label
    yaxis_title_text='',
    showlegend = False, # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1) # gap between bars of the same location coordinates
Deviceprotection = DATA.deviceprotection.value_counts()
Deviceprotection_service = [('Has Protection',Deviceprotection[0]), ('No Protection',Deviceprotection[1]),('No Internet',Deviceprotection[2])]
Deviceprotection1 = pd.DataFrame(Deviceprotection_service,columns = ['Name','Value'])

fig35 = go.Figure(go.Bar(
            x= Deviceprotection1.Name,
            y= Deviceprotection1.Value,
            marker_color='#E94B8F',
            opacity=0.75,
            orientation='v'))

fig35.update_layout(
    title_text='Number of Clients Subscribed to Device Protection', # title of plot
    xaxis_title_text='', # xaxis label
    yaxis_title_text='', # yaxis label
    bargap=0.4, # gap between bars of adjacent location coordinates
    bargroupgap=0.1) # gap between bars of the same location coordinates
Techsupport = DATA.techsupport.value_counts()
Techsupport_service = [('Has Support',Techsupport[0]), ('No Support',Techsupport[1]),('No Internet',Techsupport[2])]
Techsupport1 = pd.DataFrame(Techsupport_service,columns = ['Name','Value'])

fig36 = go.Figure(go.Bar(
            x= Techsupport1.Name,
            y= Techsupport1.Value,
            marker_color='#E94B8F',
            opacity=0.75,
            orientation='v'))

fig36.update_layout(
    title_text='Number of Clients Subscribed to Tech Support', # title of plot
    xaxis_title_text='', # xaxis label
    yaxis_title_text='', # yaxis label
    bargap=0.4, # gap between bars of adjacent location coordinates
    bargroupgap=0.1) # gap between bars of the same location coordinates
Streamingtv = DATA.streamingtv.value_counts()
Streamingtv_service = [('Has TV Streaming',Streamingtv[0]), ('No TV Streaming',Streamingtv[1]),('No Internet',Streamingtv[2])]
Streamingtv1 = pd.DataFrame(Streamingtv_service,columns = ['Name','Value'])

fig37 = go.Figure(go.Bar(
            x= Streamingtv1.Name,
            y= Streamingtv1.Value,
            marker_color='#DB2DEE',
            opacity=0.75,
            orientation='v'))

fig37.update_layout(
    title_text='Number of Clients Subscribed to TV Streaming', # title of plot
    xaxis_title_text='', # xaxis label
    yaxis_title_text='', # yaxis label
    bargap=0.4, # gap between bars of adjacent location coordinates
    bargroupgap=0.1) # gap between bars of the same location coordinates
Streamingmovies = DATA.streamingmovies.value_counts()
Streamingmovies_service = [('Has Movie Streaming',Streamingmovies[0]), ('No Movie Streaming',Streamingmovies[1]),('No Internet',Streamingmovies[2])]
Streamingmovies1 = pd.DataFrame(Streamingmovies_service,columns = ['Name','Value'])

fig38 = go.Figure(go.Bar(
            x= Streamingmovies1.Name,
            y= Streamingmovies1.Value,
            marker_color='#DB2DEE',
            opacity=0.75,
            orientation='v'))

fig38.update_layout(
    title_text='Number of Clients Subscribed to Movie Streaming', # title of plot
    xaxis_title_text='', # xaxis label
    yaxis_title_text='', # yaxis label
    bargap=0.4, # gap between bars of adjacent location coordinates
    bargroupgap=0.1) # gap between bars of the same location coordinates


###

Contracts = DATA.contract.value_counts()
con = [('1 Month',Contracts[0]), ('1 Year',Contracts[1]),('2 years',Contracts[2])]
Contracts1 = pd.DataFrame(con,columns = ['Name','Value'])
fig4 = go.Figure(go.Bar(
            y= Contracts1.Name,
            x= Contracts1.Value,
                        marker_color='#DB2DEE',
                        opacity=0.75,
                        orientation='h'))
fig4.update_layout(
    title="Number of Clients per Type of Contracts")


###payment_method

Payment = DATA.paymentmethod.value_counts()
pay = [('Electronic Check',Payment[0]), ('Mailed Check',Payment[1]),('Bank Transfer',Payment[2]),('Credit Card',Payment[3])]
Payment1 = pd.DataFrame(pay,columns = ['Name','Value'])

fig5 = px.pie(Payment1, values='Value', names='Name', hole=.3, color_discrete_sequence=px.colors.sequential.Agsunset)
fig5.update_traces(textposition='inside', textinfo='percent+label')
fig5.update_layout(title_text='Payment Methods of Clients',showlegend=False)

##General Visual Data Related to Streamlit

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {1920}px;
        padding-top: {-2}rem;
        padding-right: {5}rem;
        padding-left: {5}rem;
        padding-bottom: {5}rem;
    }}
    .reportview-container .main {{
        color: #1D3540;
        background-color: #ffffff;
    }}

</style>
"""
    ,
        unsafe_allow_html=True,
    )

##Visualizations with Streamlit
DATA12 = pd.DataFrame(DATA)

#st.beta_set_page_config(layout="wide")
st.image("https://www.axians.com/app/uploads/sites/11/2021/04/HeaderBanner_naas_news.jpg")
c01, c02, c03, c04,c05= st.beta_columns((4,1,1,1,1))
with c01:
    st.title("Telecom Churn Analysis")

c00,c1, c2, c3, c4 = st.beta_columns((4,1.2,0.7,0.7,0.7))
with c00:
        st.write("Click the buttons on the right to view the changes in each domain,you can scroll down to the model or select the churn button to predict the churn")

c0001, c0002, c0003 = st.beta_columns((1,1,1))
with c0001:
    st.write("")



with c1 :
     button1 = st.button(label = 'Contracts&Payment', key= 1)

with c2 :
    button2 = st.button(label = 'Services', key= 1)
with c3 :
    button3 = st.button(label = 'Clients', key= 1)
with c4 :
    button12 = st.button(label = 'Churn', key= 1)

if button1:
    c11,c12 = st.beta_columns((2.75,2.25))
    with c11:
       st.write(fig5)

    with c12:
       st.write(fig4)

    c110,c120 = st.beta_columns((2.75,2.25))

    with c110:
       st.write('The total payments by Electronic Checks is '+'**{}**'.format(Payment1.values[0][1]))
       st.write('The total payments by Mailed Check is '+'**{}**'.format(Payment1.values[1][1]))
       st.write('The total payments by Bank Transfer is '+'**{}**'.format(Payment1.values[2][1]))
       st.write('The total payments by Credit Cards is '+'**{}**'.format(Payment1.values[3][1]))
       st.write('')
       st.write('')
    with c120:
        st.write('It is recommended to identify the number of customers for each type of contracts as it can help in determining the loyalty of customers. It appears in the chart above that '+'**{}**'.format(Contracts.max()) +' customers are monthly subscribers, which may be an indicator of low customer loyalty.')

    c14, c15,c504, c501,c502 = st.beta_columns((1,1,1,1,1))
    with c14:
        st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTw_U1KULDZAwUR9sAc6XuGqsXIFLjtYBGCgA&usqp=CAU')

    with c15:
        st.write('**Average Spending of Customers**')
        st.write('Your average customer spends'+ ' **{}** '.format(avg_monthly) +' every month')
        st.write('Your average customer spends a total of'+ ' **{}** '.format(avg_total))
    with c501:
        st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTw_U1KULDZAwUR9sAc6XuGqsXIFLjtYBGCgA&usqp=CAU')
    with c504:
        st.write('')

    with c502:
        st.write('**Total Spending of Customers**')
        st.write('The total sum of customer spending per month is ' + '**{}** '.format(sum_monthly))
        st.write('The total sum of customer spending is '+'**{}**'.format(sum_total))

    st.write('')



if button2:
    c2300,c2400 = st.beta_columns((2,1))
    with c2300:
        st.write(fig3)
        st.write('**Note: **' + 'There are '+'**{}**'.format(Internet.max())+ ' Fiber Optic internet subscriptions. Since it is the most modern type, customers have higher expectations of the speed provided through this service. It is important to pay attention to speed quality at all times as this can be a major reason for customer churn')
        st.write('')
    with c2400:
        st.image('https://i.pinimg.com/originals/2f/a0/40/2fa040f9b49c90ab5d2807a1da567264.jpg')
    c1300,c1400 = st.beta_columns((1,1))
    with c1300:
        st.write(fig31)
    with c1400:
        st.write(fig32)
        st.write('**Note: **' + 'Keep in mind that many customers just stop using one of their lines, so the customer is not lost')



    c1301,c1401 = st.beta_columns((1,1))
    with c1301:
        st.write(fig33)
        st.write('There are  '+'**{}**'.format(Onlinesecurity.max())+ ' and '+ '**{}**'.format(Onlinebackup.max())+ ' subscribers to online backup and online security, but it is concerning that many other customers do not have online security and backup and this can cause them to churn if dissatisfied due to lower shifitng costs.')
        st.write('')
    with c1401:
        st.write(fig34)

    c131,c141 = st.beta_columns((1,1))
    with c141:
        st.write(fig35)
        st.write('There are  '+'**{}**'.format(Deviceprotection.max())+ ' and '+ '**{}**'.format(Techsupport.max())+ ' subscribers to device protection and tech support, but it is concerning that many other customers do not have device protection and tech support and this can cause them to churn if dissatisfied due to lower shifitng costs.')
        st.write('')
    with c131:
        st.write(fig36)
    c132,c142= st.beta_columns((1,1))
    with c132:
        st.write(fig37)
    with c142:
        st.write(fig38)
        st.write('**Note: **' + 'Clients subscribed to streaming services can be sensitive to internet speed')

if button3:
    c14,c16 = st.beta_columns((1,1))
    with c14:
        st.write(fig2)
    with c16:
        st.write(fig208)
        st.write('**{}**'.format(Seniorcitizen.max()) + ' of our clients are senior citizens. Since senior citizens tend to have different needs than younger generations, it is important to adjust services with respect to their level of knowledge of technology.')
    c240,c250 = st.beta_columns((1,1))
    with c240:
        st.write(fig206)
    with c250:
        st.write(fig207)

###Building the ML model





from sklearn.preprocessing import StandardScaler

train1 = DATA



y = train1.churn
X = train1

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 100)
x_train = x_train.select_dtypes(exclude=[np.number])
x_test= x_test.select_dtypes(exclude=[np.number])
x_train['contract'] = x_train['contract'].astype('category')
x_train['contract'] = x_train['contract'].cat.codes

x_test['contract'] = x_test['contract'].astype('category')
x_test['contract'] = x_test['contract'].cat.codes

x_train['gender'] = x_train['gender'].astype('category')
x_train['gender'] = x_train['gender'].cat.codes

x_test['gender'] = x_test['gender'].astype('category')
x_test['gender'] = x_test['gender'].cat.codes

x_train['internetservice'] = x_train['internetservice'].astype('category')
x_train['internetservice'] = x_train['internetservice'].cat.codes

x_test['internetservice'] = x_test['internetservice'].astype('category')
x_test['internetservice'] = x_test['internetservice'].cat.codes

x_train['multiplelines'] = x_train['multiplelines'].astype('category')
x_train['multiplelines'] = x_train['multiplelines'].cat.codes

x_test['multiplelines'] = x_test['multiplelines'].astype('category')
x_test['multiplelines'] = x_test['multiplelines'].cat.codes

x_train['seniorcitizen'] = x_train['seniorcitizen'].astype('category')
x_train['seniorcitizen'] = x_train['seniorcitizen'].cat.codes

x_test['seniorcitizen'] = x_test['seniorcitizen'].astype('category')
x_test['seniorcitizen'] = x_test['seniorcitizen'].cat.codes

x_train['onlinesecurity'] = x_train['onlinesecurity'].astype('category')
x_train['onlinesecurity'] = x_train['onlinesecurity'].cat.codes

x_test['onlinesecurity'] = x_test['onlinesecurity'].astype('category')
x_test['onlinesecurity'] = x_test['onlinesecurity'].cat.codes

x_train['techsupport'] = x_train['techsupport'].astype('category')
x_train['techsupport'] = x_train['techsupport'].cat.codes

x_test['techsupport'] = x_test['techsupport'].astype('category')
x_test['techsupport'] = x_test['techsupport'].cat.codes

x_train['onlinebackup'] = x_train['onlinebackup'].astype('category')
x_train['onlinebackup'] = x_train['onlinebackup'].cat.codes

x_test['onlinebackup'] = x_test['onlinebackup'].astype('category')
x_test['onlinebackup'] = x_test['onlinebackup'].cat.codes

x_train['deviceprotection'] = x_train['deviceprotection'].astype('category')
x_train['deviceprotection'] = x_train['deviceprotection'].cat.codes

x_test['deviceprotection'] = x_test['deviceprotection'].astype('category')
x_test['deviceprotection'] = x_test['deviceprotection'].cat.codes

x_train['streamingtv'] = x_train['streamingtv'].astype('category')
x_train['streamingtv'] = x_train['streamingtv'].cat.codes

x_test['streamingtv'] = x_test['streamingtv'].astype('category')
x_test['streamingtv'] = x_test['streamingtv'].cat.codes

x_train['streamingmovies'] = x_train['streamingmovies'].astype('category')
x_train['streamingmovies'] = x_train['streamingmovies'].cat.codes

x_test['streamingmovies'] = x_test['streamingmovies'].astype('category')
x_test['streamingmovies'] = x_test['streamingmovies'].cat.codes

x_train['paymentmethod'] = x_train['paymentmethod'].astype('category')
x_train['paymentmethod'] = x_train['paymentmethod'].cat.codes

x_test['paymentmethod'] = x_test['paymentmethod'].astype('category')
x_test['paymentmethod'] = x_test['paymentmethod'].cat.codes

x_train['partner'] = x_train['partner'].astype('category')
x_train['partner'] = x_train['partner'].cat.codes

x_test['partner'] = x_test['partner'].astype('category')
x_test['partner'] = x_test['partner'].cat.codes

x_train['phoneservice'] = x_train['phoneservice'].astype('category')
x_train['phoneservice'] = x_train['phoneservice'].cat.codes

x_test['phoneservice'] = x_test['phoneservice'].astype('category')
x_test['phoneservice'] = x_test['phoneservice'].cat.codes

x_train['dependents'] = x_train['dependents'].astype('category')
x_train['dependents'] = x_train['dependents'].cat.codes

x_test['dependents'] = x_test['dependents'].astype('category')
x_test['dependents'] = x_test['dependents'].cat.codes


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=4, random_state = 10)

model.fit(x_train,y_train)
from sklearn.metrics import accuracy_score

pred_cv = model.predict(x_train)
accuracy_score(y_train,pred_cv)



print(x_train.contract.value_counts())




def prediction(contract,internetservice,gender,seniorcitizen,partner,dependents,phoneservice,multiplelines,onlinesecurity,onlinebackup,deviceprotection,techsupport,streamingtv,streamingmovies,paymentmethod):

    # Pre-processing user input
    if contract == "two year":
            contract = 2
    elif contract == "one year":
            contract = 1
    else:
            contract = 0

    if gender == "female":
                gender = 0

    else:
                gender = 1

    if partner == "yes":
                partner = 1

    else:
                partner = 0

    if dependents == "yes":
                dependents = 1

    else:
                dependents = 0

    if seniorcitizen == "yes":
                seniorcitizen = 1

    else:
                seniorcitizen = 0


    if internetservice == "fiber optic":
                internetservice = 0
    elif internetservice == "dsl":
                internetservice = 1
    else:
                internetservice = 2


    if multiplelines == "yes":
                multiplelines = 2
    elif multiplelines == 'no':
                multiplelines = 0
    else:
                multiplelines = 1

    if onlinesecurity == "yes":
                onlinesecurity = 2
    elif onlinesecurity == 'no':
                onlinesecurity = 0
    else:
                onlinesecurity = 1

    if phoneservice == "yes":
            phoneservice = 1
    else:
            phoneservice = 0

    if onlinebackup == "yes":
                onlinebackup = 2
    elif onlinebackup == 'no':
                onlinebackup = 0
    else:
                onlinebackup = 1

    if deviceprotection == "yes":
                deviceprotection = 2
    elif deviceprotection == 'no':
                deviceprotection = 0
    else:
                deviceprotection = 1
    if techsupport == "yes":
                techsupport = 2
    elif techsupport == 'no':
                techsupport = 0
    else:
                techsupport = 1

    if streamingtv == "yes":
                streamingtv = 2
    elif streamingtv == 'no':
                streamingtv = 0
    else:
                streamingtv = 1

    if streamingmovies == "yes":
                streamingmovies = 2
    elif streamingmovies == 'no':
                streamingmovies = 0
    else:
                streamingmovies = 1

    if paymentmethod == "electronic check":
                paymentmethod = 2
    elif paymentmethod == 'mailed check':
                paymentmethod = 3
    elif paymentmethod == 'credit card (automatic)':
                    paymentmethod = 1
    elif paymentmethod == 'bank transfer (automatic)':
                paymentmethod = 0
    else :
            paymentmethod = 4





    features = np.array([contract,internetservice,gender,seniorcitizen,partner,dependents,phoneservice,multiplelines,onlinesecurity,onlinebackup,deviceprotection,techsupport,streamingtv,streamingmovies,paymentmethod])
    features = features.reshape(1, -1)
    prediction = model.predict(features)


    if prediction == 0:
                return' Not Churn'
    elif prediction == 1:
                return ' Churn'





def main():
    #front end elements of the web page
    html_temp = """
    <div style ="background-color:purple;padding:13px">
    <h1 style ="color:black;text-align:center;">Churn Prediction App</h1>
    </div>
    """


    st.markdown(html_temp, unsafe_allow_html = True)


    contract = st.selectbox('Contract',("Two Year","One Year","Monthly"))
    internetservice = st.selectbox('Internet Service',("Fiber Optic","DSL","None"))
    multiplelines = st.selectbox('More than 1 line?',("Yes","No","No Phone"))
    onlinesecurity = st.selectbox('Online Security Service',("Yes","No","No Internet"))
    gender = st.selectbox('Gender',('Male','Female'))
    seniorcitizen = st.selectbox('Senior Citizen',('Yes','No'))
    partner = st.selectbox('Has a Partner?',('Yes','No'))
    dependents = st.selectbox('Has Dependents?',('Yes','No'))
    phoneservice = st.selectbox('Has Phone Service?',("Yes","No"))
    onlinebackup = st.selectbox('Has Online Backup?',("Yes","No","No Internet"))
    deviceprotection = st.selectbox('Has device protection?',("Yes","No","No Internet"))
    techsupport = st.selectbox('Has techsupport?',("Yes","No","No Internet"))
    streamingtv = st.selectbox('TV streaming services?',("Yes","No","No Internet"))
    streamingmovies = st.selectbox('Movie streaming services?',("Yes","No","No Internet"))
    paymentmethod = st.selectbox('What payment method is used?',("Electronic Check","Mailed Check","Bank Transfer","Credit Card"))

    result = ""



    if st.button('Predict'):
        result = prediction(contract,internetservice,gender,seniorcitizen,partner,dependents,phoneservice,multiplelines,onlinesecurity,onlinebackup,deviceprotection,techsupport,streamingtv,streamingmovies,paymentmethod)
        st.success('Your customer will {}'.format(result))


if __name__=='__main__':
    main()
