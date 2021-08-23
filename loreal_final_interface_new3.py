import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from scipy.stats import iqr
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate


df = pd.read_csv('loreal_final_new.csv')

open_pickle = open('final7.obj',"rb")

pickle_load = pickle.load(open_pickle)


import plotly as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go






####Main Goals vs Brand + Product
df_plot = pd.read_csv('loreal_final_new.csv')
df_plot['brand'] = df_plot['brand'].replace({'mny': "Maybelline", 'nyx': "NYX",'oap': "L'Oreal Paris", 'gar': "Garnier",'ess': "Essie", 'mat': "Matrix",'oap-mup': "L'Oreal Makeup", 'lp': "L'Oreal Pro",'ker': "Kerastase",'lan': "Lancome", 'ysl': "Yves SL",'gio': "Giorgio Armani"})
df_plot['products'] = df_plot['products'].replace({'lash-lift': "Lash Lift", 'skin-active': "Skin Active",'mup': "Make Up", 'mass': "Mass",'care&color': "Care&Color", 'born-to-glow': "Born to Glow",'all-products': "All Products", 'fb-serum': "FB Serum",'color-naturals': "Color Naturals",'fit-me': "Fit ME"})
df_plot['audience_type'] = df_plot['audience_type'].replace({'launch': "Launch", 'custom': "Custom A",'customom': "Custom B", 'socio-int': "Socio-Int"})
df_plot['targetting'] = df_plot['targetting'].replace({'prospecting': "Prospecting", 'prospecting-non-dynamic-ads': "Prospecting (Non-Dynamic)",'retargeting': "Retargeting", 'retargeting-dynamic-ads': "Retargeting (Dynamic Ads)",'other': "Other"})
df_plot['end_month'] = df_plot['end_month'].replace({'JanuaryFebruary':'End of January',"January0":'January'})

cols7 = df_plot.groupby('brand', as_index=False)['Reach'].mean()
cols8 = df_plot.groupby('brand', as_index=False)['Impressions'].mean()
cols9 = df_plot.groupby('brand', as_index=False)['Page_Engagement'].mean()
cols10 = df_plot.groupby('brand', as_index=False)['Post Engagement'].mean()
cols11 = df_plot.groupby('brand', as_index=False)['Gross_Impressions'].mean()
cols12 = df_plot.groupby('brand', as_index=False)['Frequency'].mean()
cols13 = df_plot.groupby('brand', as_index=False)['All_Clicks'].mean()

cols15 = df_plot.groupby('products', as_index=False)['Reach'].mean()
cols16 = df_plot.groupby('products', as_index=False)['Impressions'].mean()
cols17 = df_plot.groupby('products', as_index=False)['Page_Engagement'].mean()
cols18 = df_plot.groupby('products', as_index=False)['Post Engagement'].mean()
cols19 = df_plot.groupby('products', as_index=False)['Gross_Impressions'].mean()
cols20 = df_plot.groupby('products', as_index=False)['Frequency'].mean()
cols21 = df_plot.groupby('products', as_index=False)['All_Clicks'].mean()

cols22 = df_plot.groupby('brand', as_index=False)['Amount_Spent_USD'].mean()
cols23 = df_plot.groupby('products', as_index=False)['Amount_Spent_USD'].mean()



cols71 = df_plot.groupby('end_month',as_index=False)['Reach'].max()
cols81 = df_plot.groupby('end_month',as_index=False)['Impressions'].max()

import plotly.figure_factory as ff



cols22= cols22.rename(columns= {'brand':'Brand'})
cols22= cols22.rename(columns= {'Amount_Spent_USD': 'Average Amount Spent per Brand'})


cols23 = cols23.rename(columns= {'Amount_Spent_USD': 'Average Amount Spent per Product'})
cols23 = cols23.rename(columns= {'products':'Product'})

fig22 = ff.create_table(cols22, height_constant=30, colorscale = [[0, '#B67E6F'],[.5, '#FFFFFF'],[1, '#FFFFFF']])
fig23 = ff.create_table(cols23, height_constant=30, colorscale = [[0, '#8F536B'],[.5, '#FFFFFF'],[1, '#FFFFFF']])

fig22.update_layout(
    title_text='Maximum Reach per End Month',
    xaxis_tickfont_size=14, titlefont_size = 14)

fig23.update_layout(
    title_text='Maximum Impressions per End Month',
    xaxis_tickfont_size=14, titlefont_size = 14)


fig300 = ff.create_table(cols23, height_constant=30, colorscale = [[0, '#8F536B'],[.5, '#FFFFFF'],[1, '#FFFFFF']])

fig300.update_layout(
    title_text='Maximum Reach per End Month',
    xaxis_tickfont_size=14, titlefont_size = 14)


cols301 = df_plot.groupby('targetting',as_index=False)['Amount_Spent_USD'].mean()
cols302= df_plot.groupby('audience_type',as_index=False)['Amount_Spent_USD'].mean()
cols303 = df_plot.groupby('socioint',as_index=False)['Amount_Spent_USD'].mean()

cols304 = df_plot.groupby('brand',as_index=False)['Duration_days'].mean()
cols305 = df_plot.groupby('products',as_index=False)['Duration_days'].mean()



cols301 = cols301.rename(columns = {'targetting':'Targeting Method'})
cols301 = cols301.rename(columns = {'Amount_Spent_USD':'Amount Spent (USD)'})


cols302 = cols302.rename(columns = {'audience_type':'Audience Type'})
cols302 = cols302.rename(columns = {'Amount_Spent_USD':'Amount Spent (USD)'})

cols303 = cols303.rename(columns = {'socioint':'Socioint'})
cols303 = cols303.rename(columns = {'Amount_Spent_USD':'Amount Spent (USD)'})

cols304 = cols304.rename(columns = {'brand':'Brand'})
cols304 = cols304.rename(columns = {'Duration_days':'Duration'})


cols305 = cols305.rename(columns = {'products':'Product'})
cols305 = cols305.rename(columns = {'Duration_days':'Duration'})

fig301 = ff.create_table(cols301, height_constant=30, colorscale = [[0, '#D8B691'],[.5, '#FFFFFF'],[1, '#FFFFFF']])
fig302 = ff.create_table(cols302, height_constant=30, colorscale = [[0, '#D8B691'],[.5, '#FFFFFF'],[1, '#FFFFFF']])
fig303 = ff.create_table(cols303, height_constant=30, colorscale = [[0, '#D8B691'],[.5, '#FFFFFF'],[1, '#FFFFFF']])
fig304 = ff.create_table(cols304, height_constant=30, colorscale = [[0, '#5B5854'],[.5, '#FFFFFF'],[1, '#FFFFFF']])
fig305 = ff.create_table(cols305, height_constant=30, colorscale = [[0, '#5B5854'],[.5, '#FFFFFF'],[1, '#FFFFFF']])




fig301.update_layout(
    title_text='Average Amount Spent per Targeting Method',
    xaxis_tickfont_size=14, titlefont_size = 14)

fig302.update_layout(
    title_text='Average Amount Spent per Audience Type',
    xaxis_tickfont_size=14, titlefont_size = 14)

fig303.update_layout(
    title_text='Average Amount Spent per Socio-int',
    xaxis_tickfont_size=14, titlefont_size = 14)

fig304.update_layout(
    title_text='Maximum Ad Duration per Brand',
    xaxis_tickfont_size=14, titlefont_size = 14)

fig305.update_layout(
    title_text='Maximum Ad Duration per Product',
    xaxis_tickfont_size=14, titlefont_size = 14)





fig7 = go.Figure()
fig7.add_trace((go.Bar(x= cols7.brand,
            y= cols7['Reach'],
            marker_color='#B67E6F',
            opacity=0.75,
            texttemplate='%{y:.2s}', textposition='outside')))


fig7.update_layout(
    title='Average Reach per Brand',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1,

    ))

fig7.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig7.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)


fig8 = go.Figure()
fig8.add_trace((go.Bar(x= cols8.brand,
            y= cols8['Impressions'],

            marker_color='#B67E6F',
            opacity=0.75,texttemplate='%{y:.2s}', textposition='outside')))

fig8.update_layout(
    title='Average Impressions per Brand',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig8.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig8.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig9 = go.Figure()
fig9.add_trace((go.Bar(x= cols9.brand,
            y= cols9['Page_Engagement'],

            marker_color='#B67E6F',
            opacity=0.75,texttemplate='%{x:.2s}', textposition='outside')))

fig9.update_layout(
    title='Average Page Engagement per Brand',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig9.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig9.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)






fig10 = go.Figure()
fig10.add_trace((go.Bar(x= cols10.brand,
            y= cols10['Post Engagement'],

            marker_color='#B67E6F',
            opacity=0.75,texttemplate='%{y:.2s}', textposition='outside')))

fig10.update_layout(
    title='Average Post Engagement per Brand',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig10.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig10.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig11 = go.Figure()
fig11.add_trace((go.Bar(x= cols11.brand,
            y= cols11['Gross_Impressions'],

            marker_color='#B67E6F',
            opacity=0.75,texttemplate='%{y:.2s}', textposition='outside')))
fig11.update_layout(
    title='Average Gross Impressions per Brand',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig11.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig11.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig12 = go.Figure()
fig12.add_trace((go.Bar(x= cols12.brand,
            y= cols12['Frequency'],

            marker_color='#B67E6F',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside'
)))

fig12.update_layout(
    title='Average Frequency per Brand',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig12.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig12.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig13 = go.Figure()

fig13.add_trace((go.Bar(x= cols13.brand,
            y= cols13['All_Clicks'],

            marker_color='#B67E6F',
            opacity=0.75,texttemplate='%{y:.2s}', textposition='outside'
)))

fig13.update_layout(
    title='Average Total Clicks per Brand',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig13.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig13.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig15 = go.Figure()
fig15.add_trace((go.Bar(x= cols15.products,
            y= cols15['Reach'],

            marker_color='#8F536B',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))

fig15.update_layout(
    title='Average Reach per Product',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig15.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig15.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig16 = go.Figure()
fig16.add_trace((go.Bar(x= cols16.products,
            y= cols16['Impressions'],

            marker_color='#8F536B',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))
fig16.update_layout(
    title='Average Impressions per Product',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig16.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig16.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)


fig17 = go.Figure()
fig17.add_trace((go.Bar(x= cols17.products,
            y= cols17['Page_Engagement'],

            marker_color='#8F536B',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))

fig17.update_layout(
    title='Average Page Engagement per Product',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig17.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig17.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig18 = go.Figure()
fig18.add_trace((go.Bar(x= cols18.products,
            y= cols18['Post Engagement'],

            marker_color='#8F536B',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))

fig18.update_layout(
    title='Average Post Engagement per Product',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig18.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig18.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)


fig19 = go.Figure()
fig19.add_trace((go.Bar(x= cols19.products,
            y= cols19['Gross_Impressions'],

            marker_color='#8F536B',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))
fig19.update_layout(
    title='Average Gross Impressions per Product',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig19.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig19.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig20 = go.Figure()
fig20.add_trace((go.Bar(x= cols20.products,
            y= cols20['Frequency'],

            marker_color='#8F536B',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))
fig20.update_layout(
    title='Average Frequency per Product',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig20.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig20.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig21 = go.Figure()
fig21.add_trace(go.Bar(x= cols21.products,
            y= cols21['All_Clicks'],

            marker_color='#8F536B',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside'))
fig21.update_layout(
    title='Average Total Clicks per Product',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig21.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig21.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)




tgc = df_plot.targetting.value_counts()

tg = [('Prospecting non dynamic ads',tgc[0]),('Retargeting dynamic ads',tgc[1]),('Other',tgc[2]),('Retargeting',tgc[3]),('Prospecting',tgc[4])]

tgdf = pd.DataFrame(tg, columns = ['name','value'])

fig24 = px.pie(tgdf, values='value', names='name',hole = 0.3, color_discrete_sequence=px.colors.sequential.Brwnyl)
fig24.update_traces(textposition='inside', textinfo='percent')

fig24.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')


adc = df_plot.audience_type.value_counts()

ad = [('Socio-Int',adc[0]),('Launch',adc[1]),('Custom A',adc[2]),('Custom B',adc[3])]
addf = pd.DataFrame(ad, columns = ['Audience Type', 'Value'])
fig25 = px.pie(addf, values='Value', names='Audience Type',hole = 0.3, color_discrete_sequence=px.colors.sequential.Brwnyl)
fig25.update_traces(textposition='inside', textinfo='percent+label')

fig25.update_traces(textposition='inside', textinfo='percent')

sic = df_plot.socioint.value_counts()

si = [('All',sic[0]),('Women',sic[1]),('Custom',sic[2])]
sidf = pd.DataFrame(si,columns = ['Socioint','Value'])
fig26 = px.pie(sidf, values='Value', names='Socioint',hole = 0.3, color_discrete_sequence=px.colors.sequential.Brwnyl)
fig26.update_traces(textposition='inside', textinfo='percent+label')
fig26.update_layout(title_x=0.5)

fig26.update_traces(textposition='inside', textinfo='percent')

cols27 = df_plot.groupby('brand', as_index=False)['Duration_days'].mean()
cols28 = df_plot.groupby('products', as_index=False)['Duration_days'].mean()

fig27 = go.Figure()
fig27.add_trace(go.Bar(x= cols27.brand,
            y= cols27['Duration_days'],

            marker_color='#5B5854',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside'))
fig27.update_layout(
    title='Average Ad Duration per Brand',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig27.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig27.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)




fig28 = go.Figure()
fig28.add_trace(go.Bar(x= cols28.products,
            y= cols28['Duration_days'],

            marker_color='#5B5854',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside'))
fig28.update_layout(
    title='Average Ad Duration per Product',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig28.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig28.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

cols29 = df_plot.groupby('targetting', as_index=False)['Reach'].mean()
cols30 = df_plot.groupby('targetting', as_index=False)['Impressions'].mean()
cols31 = df_plot.groupby('targetting', as_index=False)['Page_Engagement'].mean()
cols32 = df_plot.groupby('targetting', as_index=False)['Post Engagement'].mean()
cols33 = df_plot.groupby('targetting', as_index=False)['Gross_Impressions'].mean()
cols34 = df_plot.groupby('targetting', as_index=False)['All_Clicks'].mean()

cols35 = df_plot.groupby('audience_type', as_index=False)['Reach'].mean()
cols36 = df_plot.groupby('audience_type', as_index=False)['Impressions'].mean()
cols37 = df_plot.groupby('audience_type', as_index=False)['Page_Engagement'].mean()
cols38 = df_plot.groupby('audience_type', as_index=False)['Post Engagement'].mean()
cols39 = df_plot.groupby('audience_type', as_index=False)['Gross_Impressions'].mean()
cols40 = df_plot.groupby('audience_type', as_index=False)['All_Clicks'].mean()

fig29 = go.Figure()
fig29.add_trace((go.Bar(x= cols29.targetting,
            y= cols29['Reach'],

            marker_color='#D8B691',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))
fig29.update_layout(
    title='Average Reach per Targeting Method',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig29.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig29.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)




fig30 = go.Figure()
fig30.add_trace((go.Bar(x= cols30.targetting,
            y= cols30['Impressions'],

            marker_color='#D8B691',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))
fig30.update_layout(
    title='Average Impressions per Targeting Method',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig30.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig30.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)


fig31 = go.Figure()
fig31.add_trace((go.Bar(x= cols31.targetting,
            y= cols31['Page_Engagement'],

            marker_color='#D8B691',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))

fig31.update_layout(
    title='Average Page Engagement per Targeting Method',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig31.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig31.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)






fig32 = go.Figure()
fig32.add_trace((go.Bar(x= cols32.targetting,
            y= cols32['Post Engagement'],

            marker_color='#D8B691',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))
fig32.update_layout(
    title='Average Post Engagement per Targeting Method',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig32.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig32.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)




fig33 = go.Figure()
fig33.add_trace((go.Bar(x= cols33.targetting,
            y= cols33['Gross_Impressions'],

            marker_color='#D8B691',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))

fig33.update_layout(
    title='Average Gross Impressions per Targeting Method',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig33.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig33.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)


fig34 = go.Figure()
fig34.add_trace((go.Bar(x= cols34.targetting,
            y= cols34['All_Clicks'],

            marker_color='#D8B691',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))


fig34.update_layout(
    title='Average Total Clicks per Targeting Method',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig34.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig34.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)



fig35 = go.Figure()


fig35.add_trace((go.Bar(x= cols35.audience_type, y= cols35['Reach'], marker_color='#B67E6F',opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))

fig35.update_layout(
    title='Average Reach per Audience Type',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1))

fig35.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig35.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)





fig36 = go.Figure()
fig36.add_trace((go.Bar(x= cols36.audience_type,
            y= cols36['Impressions'],

            marker_color='#B67E6F',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))


fig36.update_layout(
    title='Average Impressions per Audience Type',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig36.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig36.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)



fig37 = go.Figure()
fig37.add_trace((go.Bar (x= cols37.audience_type,
            y= cols37['Page_Engagement'],

            marker_color='#B67E6F',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))

fig37.update_layout(
    title='Average Page Engagement per Audience Type',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig37.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig37.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)




fig38 = go.Figure()
fig38.add_trace((go.Bar(x= cols38.audience_type,
            y= cols38['Post Engagement'],

            marker_color='#B67E6F',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))

fig38.update_layout(
    title='Average Post Engagement per Audience Type',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig38.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig38.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)





fig39 = go.Figure()
fig39.add_trace((go.Bar(x= cols39.audience_type,
            y= cols39['Gross_Impressions'],

            marker_color='#B67E6F',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))

fig39.update_layout(
    title='Average Gross Impressions per Audience Type',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig39.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig39.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)



fig40 = go.Figure()
fig40.add_trace((go.Bar(x= cols40.audience_type,
            y= cols40['All_Clicks'],

            marker_color='#B67E6F',
            opacity=0.75,texttemplate='%{y:.2s}', textposition='outside')))

fig40.update_layout(
    title='Average Total Clicks per Audience Type',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig40.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig40.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)






cols41 = df_plot.groupby('brand', as_index=False)['All_CPC_USD'].mean()
cols42 = df_plot.groupby('brand', as_index=False)['CPM_per_1000imp_USD'].mean()
cols43 = df_plot.groupby('brand', as_index=False)['Cost_per_1000people_reached'].mean()
cols44 = df_plot.groupby('brand', as_index=False)['All_CTR'].mean()

cols45 = df_plot.groupby('products', as_index=False)['All_CPC_USD'].mean()
cols46 = df_plot.groupby('products', as_index=False)['CPM_per_1000imp_USD'].mean()
cols47 = df_plot.groupby('products', as_index=False)['Cost_per_1000people_reached'].mean()
cols48 = df_plot.groupby('products', as_index=False)['All_CTR'].mean()




fig41 = go.Figure()
fig41.add_trace((go.Bar(x= cols41.brand,
            y= cols41['All_CPC_USD'],

            marker_color='#8F536B',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))
fig41.update_layout(
    title='Average CPC per Brand',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig41.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig41.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig42 = go.Figure()
fig42.add_trace((go.Bar(x= cols42.brand,
            y= cols42['CPM_per_1000imp_USD'],

            marker_color='#8F536B',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))
fig42.update_layout(
    title='Average CPM per Brand',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig42.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig42.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig43 = go.Figure()
fig43.add_trace((go.Bar(x= cols43.brand,
            y= cols43['Cost_per_1000people_reached'],

            marker_color='#8F536B',
            opacity=0.75,texttemplate='%{y:.2s}', textposition='outside')))
fig43.update_layout(
    title='Average Total Cost per 1000 People Reached per Brand',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig43.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig43.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig44 = go.Figure()
fig44.add_trace((go.Bar(x= cols44.brand,
            y= cols44['All_CTR'],

            marker_color='#8F536B',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))
fig44.update_layout(
    title='Average CTR per Brand',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig44.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig44.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)



fig45 = go.Figure()
fig45.add_trace((go.Bar(x= cols45.products,
            y= cols45['All_CPC_USD'],

            marker_color='#B67E6F',
            opacity=0.75,texttemplate='%{y:.2s}', textposition='outside')))
fig45.update_layout(
    title='Average CPC per Product',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig45.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig45.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig46 = go.Figure()
fig46.add_trace((go.Bar(x= cols46.products,
            y= cols46['CPM_per_1000imp_USD'],

            marker_color='#B67E6F',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))
fig46.update_layout(
    title='Average CPM per Product',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig46.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig46.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig47 = go.Figure()
fig47.add_trace((go.Bar(x= cols47.products,
            y= cols47['Cost_per_1000people_reached'],

            marker_color='#B67E6F',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))

fig47.update_layout(
    title='Average Cost per 1000 People Reached per Product',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig47.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig47.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

fig48 = go.Figure()
fig48.add_trace((go.Bar(x= cols48.products,
            y= cols48['All_CTR'],

            marker_color='#B67E6F',
            opacity=0.75, texttemplate='%{y:.2s}', textposition='outside')))
fig48.update_layout(
    title='Average CTR per Product',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='',
        titlefont_size=14,
        tickfont_size=1
    ))

fig48.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig48.update_layout(barmode='stack', xaxis={'categoryorder':'total ascending'},title_x=0.5)

#####Dashboard#########


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


a1,a2,a3,a4,a5 = st.beta_columns((1,1,1,1,1))
with a3:
    st.image("https://cdn.freebiesupply.com/images/thumbs/2x/loreal-logo.png")


st.markdown("<h1 style='text-align: center; color: grey;'>Ad Spend Prediction Tool & Dashboard for L'Oreal Collaborative Ads</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>by Mehdi El Seblani, MS Business Analytics (American University of Beirut 21')</h3>", unsafe_allow_html=True)

st.write('')
st.write('')
st.write('')


st.header('General Overview On Main Metrics')
st.markdown('Scroll down for an overview on main ad metrics')
st.write(' ')
st.write(' ')
st.write(' ')

    #Start_Month =col42.selectbox('Select Start Month',['January','End of January','Februrary','March','April','May','June','September'])
    #Age_Adult = Age_All_Age_Groups = Age_Elderly = Age_MidAge_Adult = Age_Teenager = Age_Unknown = Age_Young_Adult = 0

dd1,dd2 = st.beta_columns(2)

with dd1:

    brand_select =st.selectbox('Select Metric for Brand',['Reach','Impressions','Page Engagement','Post Engagement','Gross Impressions','Clicks','Frequency'])

    if brand_select == 'Reach':
        st.write(fig7)
    if brand_select == 'Impressions':
        st.write(fig8)
    if brand_select == 'Page Engagement':
        st.write(fig9)
    if brand_select == 'Post Engagement':
        st.write(fig10)
    if brand_select == 'Gross Impressions':
        st.write(fig11)
    if brand_select == 'Clicks':
        st.write(fig12)
    if brand_select == 'Frequency':
        st.write(fig13)

with dd2:

    product_select =st.selectbox('Select Metric for Products',['Reach','Impressions','Page Engagement','Post Engagement','Gross Impressions','Clicks','Frequency'])

    if product_select == 'Reach':
        st.write(fig15)
    if product_select == 'Impressions':
        st.write(fig16)
    if product_select == 'Page Engagement':
        st.write(fig17)
    if product_select == 'Post Engagement':
        st.write(fig18)
    if product_select == 'Gross Impressions':
        st.write(fig19)
    if product_select == 'Clicks':
        st.write(fig20)
    if product_select == 'Frequency':
        st.write(fig21)



d071,d081 = st.beta_columns(2)

with d071:

    st.write(fig22)

with d081:

    st.write(fig23)


###Targeting###


st.header('General Overview On Targeting Details')
st.markdown('Scroll down for an overview on targeting details')
st.write(' ')
st.write(' ')
st.write(' ')

e01,e02 = st.beta_columns(2)

with e01:

    ad_dist_select = st.selectbox('Select Metric for Ad Distribution',['Targeting','Audience Type','Socio-Int'])

    if ad_dist_select == 'Targeting':
        st.markdown('Percentages of Ad Distribution per Targeting Method')
        st.write(fig24)
        st.write(fig301)
    if ad_dist_select == 'Audience Type':
        st.markdown('Percentages of Ad Distribution per Audience Type')
        st.write(fig25)
        st.write(fig302)
    if ad_dist_select == 'Socio-Int':
        st.markdown('Percentages of Ad Distribution per Socio-Int')
        st.write(fig26)
        st.write(fig303)

with e02:

    ad_dur_select = st.selectbox('Select Metric for Ad Duration',['Brand','Product'])

    if ad_dur_select == 'Brand':
        st.write(fig27)
        st.write(fig304)
    if ad_dur_select == 'Product':
        st.write(fig28)
        st.write(fig305)

ee01,ee02 = st.beta_columns(2)


with ee01:

    brand_select_target =st.selectbox('Select Metric for Brands',['Reach','Impressions','Page Engagement','Post Engagement','Gross Impressions','Clicks'])



    if brand_select_target == 'Reach':
        st.write(fig29)
    if brand_select_target == 'Impressions':
        st.write(fig30)
    if brand_select_target == 'Page Engagement':
        st.write(fig31)
    if brand_select_target == 'Post Engagement':
        st.write(fig32)
    if brand_select_target == 'Gross Impressions':
        st.write(fig33)
    if brand_select_target == 'Clicks':
        st.write(fig34)


with ee02:

    product_select_target =st.selectbox('Select Metric for Audience Type',['Reach','Impressions','Page Engagement','Post Engagement','Gross Impressions','Clicks'])



    if product_select_target == 'Reach':
        st.write(fig35)
    if product_select_target == 'Impressions':
        st.write(fig36)
    if product_select_target == 'Page Engagement':
        st.write(fig37)
    if product_select_target == 'Post Engagement':
        st.write(fig38)
    if product_select_target == 'Gross Impressions':
        st.write(fig39)
    if product_select_target == 'Clicks':
        st.write(fig40)



st.header('General Overview On Costs/Expenditures')
st.markdown('Scroll down for an overview on costs and expenditures')
st.write(' ')
st.write(' ')
st.write(' ')

f1,f2 = st.beta_columns(2)
with f1:



    brand_select_cost =st.selectbox('Select Cost Metric for Brand',['CPC','CPM(Impressions)','CPM(Reach)','CTR'])


    if brand_select_cost == 'CPC':
        st.write(fig41)
    if brand_select_cost == 'CPM(Impressions)':
        st.write(fig42)
    if brand_select_cost == 'CPM(Reach)':
        st.write(fig43)
    if brand_select_cost == 'CTR':
        st.write(fig44)
with f2:

    product_select_cost =st.selectbox('Select Cost Metric for Product',['CPC','CPM(Impressions)','CPM(Reach)','CTR'])


    if product_select_cost == 'CPC':
        st.write(fig45)
    if product_select_cost == 'CPM(Impressions)':
        st.write(fig46)
    if product_select_cost == 'CPM(Reach)':
        st.write(fig47)
    if product_select_cost == 'CTR':
        st.write(fig48)



########################################################################################################################################################################
########################################################################################################################################################################
##########################################                              Prediction Tool                                       ##########################################
########################################################################################################################################################################
########################################################################################################################################################################
st.write(' ')
st.write(' ')
st.write(' ')
st.markdown("<h2 style='text-align: center; color: dark grey;'>Ad Spend Prediction Tool</h2>", unsafe_allow_html=True)

st.write(' ')
st.write(' ')
st.write(' ')

st.subheader('Ad Goals')
col1,col2,col3,col4 =st.beta_columns(4)
reach=col1.number_input('Reach',value=1,min_value=1,max_value=30000,step=1)
impressions=col2.number_input('Impressions*',value=1,min_value=1,max_value=1200000,step=1)
Gross_Impressions = col3.number_input('Gross Impressions*',value=1,min_value=1,max_value=1200000,step=1)
frequency=col4.number_input('Frequency',value=0.01,min_value=0.01,max_value=50.00,step=0.05)



Duration = st.slider('Select Range of Days',1.0, 280.0)


st.subheader('Clicks Metrics')
col5,col6,col7,col8 =st.beta_columns(4)
all_Clicks=col5.number_input('Clicks (All)',value=1,min_value=1,max_value=330000,step=1)
link_clicks=col6.number_input('Clicks',value=1,min_value=1,max_value=175000,step=1)
unique_link_clicks =  col7.number_input('Unique Link Clicks',value=1,min_value=1,max_value=66000,step=1)
Outbound_Clicks = col8.number_input('Outbound Clicks',value=1,min_value=1,max_value=75000,step=1)

col9,col10 =st.beta_columns(2)
Unique_Outbound_Clicks = col9.number_input('Unique Outbound Clicks*',value=1,min_value=1,max_value=36000,step=1)
Unique_Clicks_All = col10.number_input('Unique Clicks (All)',value=1,min_value=1,max_value=80000,step=1)

auto_Refresh_Impressions = 0





st.subheader('Engagement Metrics')
col12,col13 =st.beta_columns(2)

Page_Engagement =  col12.number_input('Page Engagement',value=1,min_value=1,max_value=500000,step=1)
Post_Engagement = col13.number_input('Post Engagement',value=1,min_value=1,max_value=500000,step=1)


###scaling the numerical inputs
reach_scaled= (reach - 231.500000)/(1682.000000- 27.000000)
impressions_scaled = (impressions  - 365.500000)/(2493.000000 - 51.000000)
link_clicks_scaled = (link_clicks - 4.000000 )/(23.000000-0)
frequency_scaled = (frequency - 1.455423)/(2.032269 - 1.235939)

all_clicks_scaled = (all_Clicks -7.000000 )/(41.000000-1.000000)

gross_impressions_scaled = (Gross_Impressions - 366.000000)/(2502.250000 - 51.000000)


page_engagement_scaled = (Page_Engagement - 6.000000)/(33.000000 - 1.000000)
post_engagement_scaled = (Post_Engagement - 6.000000)/(33.000000 - 1.000000)


unique_link_clicks_scaled = (unique_link_clicks - 4.000000)/(21.000000 - 0.000000)
outbound_clicks_scaled = (Outbound_Clicks - 3.000000)/(13.000000 - 0.000000)

Unique_Outbound_Clicks_scaled = (Unique_Outbound_Clicks - 2.000000 )/(11.000000)

Unique_Clicks_All_scaled = (Unique_Clicks_All - 5.000000 )/(29.000000 - 1.000000)

Duration_days_scaled = (Duration - 70.000000)/(112.000000 - 54.000000)




Age_Adult = Age_All_Age_Groups = Age_Elderly = Age_MidAge_Adult = Age_Teenager = Age_Unknown = Age_Young_Adult = 0


st.subheader('Audiences and Brands')


col31,col32=st.beta_columns(2)
Age=col31.selectbox('Select Age Group',['Teenager','Young Adult','Middle Age Adult','Adult','Senior','All','Unknown'])
if Age == "Teenager":
        Age_Teenager =1

elif Age =='Young Adult':
        Age_Young_Adult=1

elif Age =='Middle Age Adult':
        Age_MidAge_Adult=1

elif Age =='Adult':
        Age_Adult=1

elif Age =='Senior':
        Age_Elderly=1

elif Age =='All':
        Age_All_Age_Groups=1

elif Age =='Unknown':
        Age_Unknown = 1


Gender_All_Genders = Gender_Unknown = Gender_female = Gender_male = 0



Gender =col32.selectbox('Select Gender',['All','Male','Female','Unknown'])
if Gender == "All":
        Gender_All_Genders = 1

elif Gender =='Unknown':
        Gender_Unknown = 1

elif Gender =='Female':
        Gender_female = 1

elif Age =='Male':
        Gender_male = 1


Ad_Set_Delivery_active = Ad_Set_Delivery_completed = Ad_Set_Delivery_inactive = Ad_Set_Delivery_not_delivering = Ad_Set_Delivery_recently_completed = 0

col33,col34=st.beta_columns(2)

Ad_Set_Delivery =col33.selectbox('Select Ad Set Delivery',['Active','Completed','Inactive','Not Delivering','Recently Completed'])

if Ad_Set_Delivery == "Active":
        Ad_Set_Delivery_active = 1

elif Ad_Set_Delivery =='Completed':
        Ad_Set_Delivery_completed = 1

elif Ad_Set_Delivery == "Inactive":
        Ad_Set_Delivery_inactive = 1

elif Ad_Set_Delivery =='Not Delivering':
        Ad_Set_Delivery_not_delivering = 1

elif Ad_Set_Delivery == "Recently Completed":
        Ad_Set_Delivery_recently_completed = 1



Attribution_Setting_28_day_click_or_1_day_view = Attribution_Setting_7_day_click = Attribution_Setting_7_day_click_or_1_day_view = 0

Attribution_Setting =col34.selectbox('Select Attribution Settings',['28 day click or 1 day view','7 day click','7 day click or 1 day view'])

if Attribution_Setting == "28 day click or 1 day view":
        Attribution_Setting_28_day_click_or_1_day_view = 1

elif Attribution_Setting =='7 day click':
        Attribution_Setting_7_day_click = 1

elif Attribution_Setting =='7 day click or 1 day view':
        Attribution_Setting_7_day_click_or_1_day_view  = 1



col35,col36=st.beta_columns(2)

brand_ess=brand_gar=brand_gio=brand_ker=brand_lan=brand_lp=brand_mat=brand_mny=brand_nyx=brand_oap=brand_oap_mup=brand_ysl = 0

Brand =col35.selectbox('Select Brand',['Essie', 'Garnier', 'Giorgio Armani','Kerastase','Lancome','Loreal Pro','Matrix','Maybelline','NYX','Loreal Paris','Loreal Paris Makeup','Yves Saint-Laurent'])

if Brand == "Essie":
        brand_ess = 1

elif Brand =='Garnier':
        brand_gar = 1

elif Brand =='Giorgio Armani':
        brand_gio = 1

elif Brand =='Kerastase':
        brand_ker = 1

elif Brand =='Lancome':
        brand_lan = 1

elif Brand =='Loreal Pro':
        brand_lp = 1

elif Brand =='Matrix':
        brand_mat = 1

elif Brand =='Maybelline':
        brand_mny = 1

elif Brand =='NYX':
        brand_nyx = 1

elif Brand =='Loreal Paris':
        brand_oap = 1

elif Brand =='Loreal Paris Makeup':
        brand_oap_mup = 1

elif Brand =='Yves Saint-Laurent':
        brand_ysl = 1

products_all_products=products_born_to_glow=products_care_color=products_color_naturals=products_fb_serum=products_fit_me=products_lash_lift=products_mass=products_mup=products_skin_active = 0

Products =col36.selectbox('Select Product',['All', 'Born to Glow', 'Care & Color','Color Naturals','FB Serum','Fit ME','Lash Lift','Mass','Make Up','Skin Active'])

if Products == "All":
        products_all_products = 1

elif Products =='Born to Glow':
        products_born_to_glow = 1

elif Products =='Care & Color':
        products_care_color = 1

elif Products =='Color Naturals':
        products_color_naturals = 1

elif Products =='FB Serum':
        products_fb_serum = 1

elif Products =='Fit ME':
        products_fit_me = 1

elif Products =='Lash Lift':
        products_lash_lift = 1

elif Products =='Mass':
        products_mass = 1

elif Products =='Make Up':
        products_mup = 1

elif Products =='Skin Active':
        products_skin_active = 1


col37,col38=st.beta_columns(2)

audience_type_custom = audience_type_customom =  audience_type_launch = audience_type_socio_int = 0
Audience =col37.selectbox('Select Audience',['Custom A','Custom B' ,'Launch', 'Socio-Int'])

if Audience == "Custom A":
        audience_type_custom = 1


elif Audience =='Custom B':
        audience_type_customom = 1

elif Audience =='Launch':
        audience_type_launch = 1

elif Audience =='Socio-Int':
        audience_type_socio_int = 1



socioint_all=socioint_socio_int=socioint_wom=0

socioint =col38.selectbox('Select Socioint',['All','Standard Socio Int' ,'Women'])

if socioint == "All":
        socioint_all = 1

elif socioint =='Standard Socio Int':
        socioint_socio_int = 1

elif socioint =='Women':
        socioint_wom = 1




col39,col40 = st.beta_columns(2)
targetting_other =  targetting_prospecting = targetting_prospecting_nondynamicads =  targetting_retargeting = targetting_retargeting_dynamicads = 0



Targetting =col39.selectbox('Select Targetting',['Prospecting','Prospecting Non-Dynamic Ads' ,'Retargetting','Retargetting Dynamic Ads','Other'])

if Targetting =='Prospecting':
        targetting_prospecting = 1

elif Targetting =='Prospecting Non-Dynamic Ads':
        targetting_prospecting_nondynamicads = 1

elif Targetting =='Retargetting':
        targetting_retargeting = 1

elif Targetting =='Retargetting Dynamic Ads':
        targetting_retargeting_dynamicads = 1

elif Targetting == "Other":
        targetting_other = 1


end_year_2020 = end_year_2021 = 0
start_year_2020 = start_year_2021 = 0



st.header('Select Previous Years with Similar Market Conditions')

col41,col42 = st.beta_columns(2)
Start_Year =col41.selectbox('Select Start Year',['2020','2021'])

if Start_Year =='2020':
        start_year_2020 = 1

elif Start_Year =='2021':
        start_year_2020 = 1


End_Year =col42.selectbox('Select End Year',['2020','2021'])

if End_Year =='2020':
        end_year_2020 = 1

elif End_Year =='2021':
        end_year_2021 = 1


end_month_April =  end_month_February =  end_month_January0 = end_month_JanuaryFebruary =  end_month_July =  end_month_June = end_month_May =0

start_month_April  =  start_month_February  =  start_month_January0  = start_month_JanuaryJanuary  =  start_month_June  =  start_month_March  = start_month_May  =  start_month_September = 0

col42,col43 = st.beta_columns(2)
Start_Month =col42.selectbox('Select Start Month',['January','End of January','Februrary','March','April','May','June','September'])

if Start_Month =='January':
        start_month_January0 = 1
elif Start_Month =='End of January':
        start_month_JanuaryJanuary = 1
elif Start_Month =='Februrary':
        start_month_February = 1
elif Start_Month =='March':
        start_month_March = 1
elif Start_Month =='April':
        start_month_April = 1
elif Start_Month =='May':
        start_month_May = 1
elif Start_Month =='June':
        start_month_June = 1
elif Start_Month =='September':
        start_month_September = 1


End_Month =col43.selectbox('Select End Month',['January','End of January','Februrary','April','May','June','July'])

if End_Month =='January':
        end_month_January0 = 1
elif End_Month =='End of January':
        end_month_JanuaryFebruary = 1
elif End_Month =='Februrary':
        end_month_February = 1
elif End_Month =='April':
        end_month_April = 1
elif End_Month =='May':
        end_month_May = 1
elif End_Month =='June':
        end_month_June = 1
elif End_Month =='July':
        end_month_July = 1


predict_button=st.button(label = 'Predict Ad Spend')
if predict_button:
        prediction = pickle_load.predict([[reach_scaled,impressions_scaled,link_clicks_scaled,frequency_scaled,all_clicks_scaled,gross_impressions_scaled,auto_Refresh_Impressions,page_engagement_scaled,post_engagement_scaled,unique_link_clicks_scaled,outbound_clicks_scaled,Unique_Outbound_Clicks_scaled,Unique_Clicks_All_scaled,Duration_days_scaled,Age_Adult,Age_All_Age_Groups,Age_Elderly,Age_MidAge_Adult ,Age_Teenager,Age_Unknown,
                                 Age_Young_Adult,Gender_All_Genders, Gender_Unknown, Gender_female, Gender_male,
                                 Ad_Set_Delivery_active,Ad_Set_Delivery_completed,Ad_Set_Delivery_inactive,
                                 Ad_Set_Delivery_not_delivering,Ad_Set_Delivery_recently_completed,
                                 Attribution_Setting_28_day_click_or_1_day_view ,
                                 Attribution_Setting_7_day_click ,  Attribution_Setting_7_day_click_or_1_day_view,
                                 brand_ess,brand_gar,brand_gio,brand_ker,brand_lan,brand_lp,brand_mat,brand_mny,brand_nyx,
                                 brand_oap,brand_oap_mup,brand_ysl,products_all_products,products_born_to_glow,
                                 products_care_color,products_color_naturals,products_fb_serum,products_fit_me,
                                 products_lash_lift,products_mass,products_mup,products_skin_active, audience_type_custom ,
                                 audience_type_customom ,  audience_type_launch , audience_type_socio_int ,
                                 socioint_all, socioint_socio_int,socioint_wom, targetting_other ,
                                 targetting_prospecting , targetting_prospecting_nondynamicads ,
                                 targetting_retargeting , targetting_retargeting_dynamicads,end_year_2020,
                                 end_year_2021,end_month_April ,  end_month_February ,  end_month_January0 ,
                                 end_month_JanuaryFebruary ,  end_month_July ,  end_month_June , end_month_May ,
                                 start_year_2020,start_year_2021, start_month_April ,  start_month_February ,
                                 start_month_January0 , start_month_JanuaryJanuary ,  start_month_June ,
                                 start_month_March , start_month_May ,  start_month_September]])

        actual_value = prediction-0.128

        st.subheader(f"The predicted Ad Spend for Collaborative Ads on Facebook Ads is between **{round(actual_value[0],3)*100}** and **{round(prediction[0],3)*100}** cents")

        col200,col201,col202,col203 = st.beta_columns(4)


        col204,col205,col206,col207 = st.beta_columns(4)
        cpm = 1000*(prediction/impressions)
        cpmr = 1000*(prediction/reach)
        cpc = (prediction/all_Clicks)
        ctr = (all_Clicks/impressions)
        cpage = (prediction/Page_Engagement)
        cpost = (prediction/Post_Engagement)

        st.write(f'Predict CPM: **{round(cpm[0],2)}** USD')
        st.write(f'Predict CPM Reach: **{round(cpmr[0],2)}** USD')
        st.write(f'Predict CPC: **{round(cpc[0],2)}** USD')
        st.write(f'Predicted CTR: **{ctr}**')
        st.write(f'Predicted Cost Per Page Engagement: **{round(cpage[0],2)}** USD')
        st.write(f'Predicted Cost Per Post Engagement: **{round(cpost[0],2)}** USD')



st.write(' ')
st.write(' ')
st.write(' ')

st.subheader("Ad Spend Calculator")
st.markdown('Input the total budget for Collab Ads:')
budget = st.number_input('Budget for Collaborative Ads in USD Cents:',value=0.00,min_value=0.00,max_value=1000.00,step=0.01)


st.write(' ')
st.write(' ')
st.markdown('Copy the predicted ad spend above and paste into the slots below to calculate **the total ad spend** and **the remaining budget**')


st.write(' ')
st.write(' ')
col46,col47,col48,col49 = st.beta_columns(4)
campaign1 =  col46.number_input('Collab Ad 1 Ad Spend:',value=0.00,min_value=0.00,max_value=10.00,step=0.01)
campaign2 = col47.number_input('Collab Ad 2 Ad Spend:',value=0.00,min_value=0.00,max_value=10.00,step=0.01)
campaign3 = col48.number_input('Collab Ad 3 Ad Spend:',value=0.00,min_value=0.00,max_value=10.00,step=0.01)
campaign4 =  col49.number_input('Collab Ad 4 Ad Spend:',value=0.00,min_value=0.00,max_value=10.00,step=0.01)

col50,col51,col52,col53 = st.beta_columns(4)
campaign5 = col50.number_input('Collab Ad 5 Ad Spend:',value=0.00,min_value=0.00,max_value=1000.00,step=0.01)
campaign6 = col51.number_input('Collab Ad 6 Ad Spend:',value=0.00,min_value=0.00,max_value=1000.00,step=0.01)
campaign7 = col52.number_input('Collab Ad 7 Ad Spend:',value=0.00,min_value=0.00,max_value=1000.00,step=0.01)
campaign8 = col53.number_input('Collab Ad 8 Ad Spend:',value=0.00,min_value=0.00,max_value=1000.00,step=0.01)




calculate_button = st.button(label = 'Calculate')
if calculate_button:
    total_campaign_spend = campaign1 + campaign2 + campaign3 + campaign4 + campaign5 + campaign6 + campaign7 + campaign8
    remaining_budget = budget - (total_campaign_spend)

    st.subheader(f"The total Ad Spend for the selected collab ads is: **{round(total_campaign_spend,2)}** cents")
    st.subheader(f"Remaining budget for collab ads is: **{round(remaining_budget,2)}** cents")

        ########################################################################################################
        ########################################    #     #   #       #   ######################################
        ##Dashboard#############################    #     #   # #   # #   ######################################
        ########################################    #     #   #   #   #   ######################################
        ## L'Oreal ##############################   #######   #       #   ######################################
        ## Collab Ads ##########################################################################################
