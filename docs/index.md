
## GNSS-based TEC anomaly detection



The changes in the ionosphere layer before, during, or after the earthquake have long been an earthquke precursor among scientists. The anomalies in Total Electron Content (TEC) of the ionosphere layer before, during, or after have been reoported by scholars. 
A wide variety of methods have been utilized for detecting anomalies in TEC time series. Although the methods have different mechanisms, all of them construct a normal profile of TEC initially, then identify values that do not conform to the normal profile as anomalies. Regarding the ionosphere nature that is always affected by different factors, finding a normal trend is a complex even impossible task. Thus, anomaly detection results have probably too many false alarms (having normal values identified as anomalies).

Here, we identify anomalies regarding the fact that anomalies are very different and rare. We calculate the TEC anomoalies using an unsupervised  machine learning algorithm, isolation forest.


![alt text](https://github.com/alikhosravi/BBox/blob/main/docs/libs.jpg?raw=true)


If you are going to run the codes on your local machine follow the code blocks presented in this page:

### Download the rquired data
To begin with, download the required <a href="https://github.com/alikhosravi/Ionosphere-GNSS-TEC/raw/main/docs/data/aruc.zip" download>data</a>. This is a .zip file containing vertical TEC of 'aruc' GNSS station from 2016/02/05 to 2016/02/25. 

Extract the files in the .zip file into a new folder. 

### Import the rquired libraries
Make a new blank .py file. Then import the rquired libraries as following:

```markdown
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import requests 
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta, date
from bs4 import BeautifulSoup
```
### Make a date range
In this section, we convert the dates that we are going to study in a day of year format.

```markdown
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

#Station('Enter staion name')
Station = 'aruc'
#'Enter start date in YY/MM/DD format'
stdate='2016/02/05'
#'Enter End date in YY/MM/DD format'
endate='2016/02/25'
dates=[]
start_dt = date(int(stdate.split('/')[0]), int(stdate.split('/')[1]), int(stdate.split('/')[2]))
end_dt = date(int(endate.split('/')[0]), int(endate.split('/')[1]), int(endate.split('/')[2]))
for dt in daterange(start_dt, end_dt):
    dates.append(dt.strftime("%Y-%m-%d"))
```

### Read data
As the next step, our program reads the 'aruc' station data. 

```markdown
Time=[];
VTec=[];
for date in dates:
    DOY= datetime.datetime(int(date.split('-')[0]), int(date.split('-')[1]), int(date.split('-')[2])).timetuple().tm_yday
    file = open(Station+'_'+date.split('-')[0]+date.split('-')[1]+date.split('-')[2]+'.txt', 'r')
    date.split('-')[0]
    while True:
        line = file.readline()
        if not line:
            break
        if line[0]!='#':
            if float(line.split(' ')[1])!=999.99:
              Time.append(DOY+(int(line.split(' ')[0].split('T')[1][0:2])/24)+(int(line.split(' ')[0].split('T')[1][2:4])/60/24)+(int(line.split(' ')[0].split('T')[1][4:6])/60/60/24))
              VTec.append(float(line.split(' ')[1]))
    file.close()


data=[]
lst=0;
for date in dates:
  rd=[];
  file = open(Station+'_'+date.split('-')[0]+date.split('-')[1]+date.split('-')[2]+'.txt', 'r')
  while True:
    line = file.readline()
    if not line:
      break
    if line[0]!='#':
      if float(line.split(' ')[1])!=999.99:
        data.append([datetime.datetime(int(line.split(' ')[0][:4]), int(line.split(' ')[0][4:6]), int(line.split(' ')[0][6:8])).timetuple().tm_yday,
                      line.split(' ')[0][9:11],
                      line.split(' ')[0][11:13],
                      line.split(' ')[0][13:15],
                      float(line.split(' ')[1])]);

  file.close()
```  
### Make a dataframe
We create a Pandas dataframe to store our data and work on it.

```markdown
df = pd.DataFrame(data, columns=['Day', 'Hour', 'Minute', 'Second', 'VTEC'])
vTecs=[];
ldays=[];
for day in list(df.groupby('Day')):
  vTecs.append(list(day[1]['VTEC']))
  ldays.append(day[0])
vr = list(map(list, zip(*vTecs)))
NewDF = pd.DataFrame(vr, columns=ldays)
```

### Identify anomalies
In this part we detect anomalies based on the isolation forest algorithm.

```markdown
Outliers_x=[]
Outliers_y=[]
count=0;
def IF(List):
  global count
  RList=List
  List= np.array(List).reshape(-1, 1)
  model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.01),max_features=1.0)
  model.fit(List)
  model.decision_function(List)
  output = model.predict(List)

  try:
    list(output).index(-1);
    indices = [[i+ldays[0]+(count/2/60/24), list(RList)[i]] for i, x in enumerate(output) if x == -1];
    for r in indices:
      Outliers_x.append(r[0]);
      Outliers_y.append(r[1]);

  except:
    return 1
  count+=1;

df1 = NewDF.apply(lambda row : IF(row), axis = 1)
```

```markdown
Response = requests.get('https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt');
with open("KP_data.txt", "w") as f:
    f.write(Response.text)

KP_data=[]
KP_Values=[];
KP_Times=[];
F10s =[];
F10_Times=[];
KP_columns = ['YYY','MM','DD','days','days_m','Bsr','dB','Kp1','Kp2','Kp3','Kp4','Kp5','Kp6','Kp7','Kp8','ap1','ap2','ap3','ap4','ap5','ap6','ap7','ap8','Ap','SN','F10.7obs','F10.7adj','D']
file = open('KP_data.txt', 'r')
while True:
  line = file.readline()
  if not line:
    break
  if line[0]!='#':
   KP_data.append(list(filter(None, line.split(' '))))
Kp_df=pd.DataFrame(KP_data, columns= KP_columns)
for date in dates:
  DOY= datetime.datetime(int(date.split('-')[0]), int(date.split('-')[1]), int(date.split('-')[2])).timetuple().tm_yday
  KPs = list(Kp_df.loc[(Kp_df['YYY'] == date.split('-')[0]) & (Kp_df['MM'] == date.split('-')[1]) & (Kp_df['DD'] == date.split('-')[2])][['Kp1',	'Kp2',	'Kp3','Kp4',	'Kp5',	'Kp6', 'Kp7', 'Kp8']].values[0])
  F10s.append(float(Kp_df.loc[(Kp_df['YYY'] == date.split('-')[0]) & (Kp_df['MM'] == date.split('-')[1]) & (Kp_df['DD'] == date.split('-')[2])][['F10.7obs']].values[0][0]))
  F10_Times.append(DOY)
  time = 3/24
  for value in KPs:
    KP_Values.append(float(value))
    KP_Times.append(DOY+time)
    time+=(3/24)


Months=[]
for date in dates:
  Months.append(date.split('-')[0]+date.split('-')[1]);
Months = list(set(Months))
Dst_df = pd.DataFrame()
for month in Months:
  if int(month[0:4])>2016:
    response = requests.get('https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/'+month+'/index.html');
  else:
    response = requests.get('https://wdc.kugi.kyoto-u.ac.jp/dst_final/'+month+'/index.html');
  with open(month+".html", "w") as f:f.write(response.text)
  with open(month+".html") as fp:soup = BeautifulSoup(fp, "html.parser")
  cnt = soup.find("pre", class_="data")
  List_lines = cnt.text.split('\n')
  List_lines = list(filter(None, List_lines))
  List_lines = List_lines[List_lines.index('DAY')+1:]
  for l in List_lines:
    x = list(filter(None, l.split(' ')))
    x[0] = month+x[0]
    try:
      Dst_df[x[0]]=x[1:]
    except:
      print(x[0], x[1:])

Dst_Times=[];
Dst_Values=[];

for date in dates:
  DOY = datetime.datetime(int(date.split('-')[0]), int(date.split('-')[1]), int(date.split('-')[2])).timetuple().tm_yday;
  DDst = list(map(int, list(Dst_df[date.split('-')[0]+date.split('-')[1]+str(int(date.split('-')[2]))])))
  dly = 1/24
  for dd in DDst:
    Dst_Times.append(DOY+dly)
    Dst_Values.append(dd)
    dly+=(1/24)



if len(dates)<100:
  if len(dates)<10:
    Dur_Day_Tens = '00';
    Dur_Day=str(len(dates))
  else:
    Dur_Day_Tens = '0'+str(len(dates))[0];
    Dur_Day = str(len(dates))[1];
else:
  Dur_Day_Tens = str(len(dates))[:2];
  Dur_Day = str(len(dates))[2];

url = 'https://wdc.kugi.kyoto-u.ac.jp/cgi-bin/aeasy-cgi?Tens='+stdate.split('/')[0][:3]+'&Year='+stdate.split('/')[0][3]+'&Month='+stdate.split('/')[1]+'&Day_Tens='+stdate.split('/')[2][0]+'&Days='+stdate.split('/')[2][1]+'&Hour=00&min=00&Dur_Day_Tens='+Dur_Day_Tens+'&Dur_Day='+Dur_Day+'&Dur_Hour=00&Dur_Min=00&Image+Type=GIF&COLOR=COLOR&AE+Sensitivity=0&ASY%2FSYM++Sensitivity=0&Output=AE&Out+format=IAGA2002'

AE_res = requests.get(url)
with open("AE_data.txt", "w") as f:f.write(AE_res.text)
file = open('AE_data.txt', 'r')
AE_data=[]
while True:
  line = file.readline()
  if not line:
    break
  if line[0]!='#':
   AE_data.append(list(filter(None, line.split(' '))))
AE_Values=[];
AU_Values=[];
AL_Values=[];
AO_Values=[]
A_Times=[]
try:
  AE_data = AE_data[AE_data.index(['DATE', 'TIME', 'DOY', 'AE', 'AU', 'AL', 'AO', '|\n'])+1:-2]
  for data in AE_data:
    try:
      AE_Values.append(float(data[3]));
      AU_Values.append(float(data[4]));
      AL_Values.append(float(data[5]));
      AO_Values.append(float(data[6][:-2]));
      A_Times.append(int(data[2])+float(int(data[1].split(':')[1])/60/24)+float(int(data[1].split(':')[0])/24));
    except:
      pass
except:
  pass

S_DOY= int(datetime.datetime(int(dates[0].split('-')[0]), int(dates[0].split('-')[1]), int(dates[0].split('-')[2])).timetuple().tm_yday)
E_DOY= int(datetime.datetime(int(dates[-1].split('-')[0]), int(dates[-1].split('-')[1]), int(dates[-1].split('-')[2])).timetuple().tm_yday)
EQ_DOY=312
fig,axes=plt.subplots(nrows=4,ncols=1,figsize=(60,48))
matplotlib.rcParams['axes.linewidth'] = 2

#VTEC and Outliers
axes[0].plot(Time, VTec, linewidth=3.0, label = 'VTEC', color= 'blue')
axes[0].plot(Outliers_x, Outliers_y,'o', label = 'Anomalies', color='orange')
axes[0].set_xticks(np.linspace(S_DOY,E_DOY,(E_DOY-S_DOY+1)))
#axes[0].axvline(x = EQ_DOY, color = 'black', linestyle='--', label = 'EQ occurrence', linewidth=3.0)
axes[0].legend(loc="upper left", prop={'size': 30})
axes[0].set_ylabel("TECU", fontsize=30)
axes[0].tick_params(axis='both', which='major', labelsize=30)
axes[0].tick_params(axis='both', which='minor', labelsize=30)


#KP
axes[1].plot(KP_Times,KP_Values, linewidth=3.0, label = 'Kp index', color= 'blue')
axes[1].set_xticks(np.linspace(S_DOY,E_DOY,(E_DOY-S_DOY+1)))
axes[1].axhline(y =3 , color = 'r', linewidth=3.0, linestyle='--', label = 'Geomagnetic storm threshold')
axes[1].legend(loc="upper left", prop={'size': 30})
axes[1].tick_params(axis='both', which='major', labelsize=30)
axes[1].tick_params(axis='both', which='minor', labelsize=30)


#10.7
axes[2].bar(F10_Times,F10s, linewidth=3.0, label = 'F10.7', color= 'blue')
axes[2].set_xticks(np.linspace(S_DOY,E_DOY,(E_DOY-S_DOY+1)))
axes[2].legend(loc="upper left", prop={'size': 30})
axes[2].tick_params(axis='both', which='major', labelsize=30)
axes[2].tick_params(axis='both', which='minor', labelsize=30)

#Dst
axes[3].plot(Dst_Times, Dst_Values,linewidth=3.0, label='Dst' , color= 'blue')
axes[3].set_xticks(np.linspace(S_DOY,E_DOY,(E_DOY-S_DOY+1)))
axes[3].axhline(y =-20 , color = 'r', linestyle='--', linewidth=3.0, label = 'Geomagnetic storm threshold')
axes[3].legend(loc="upper left", prop={'size': 30})
axes[3].set_ylabel("nT",fontsize=30)
axes[3].tick_params(axis='both', which='major', labelsize=30)
axes[3].tick_params(axis='both', which='minor', labelsize=30)
axes[3].set_xlabel("Time (DOY)" ,fontsize=30)
try:
  #AE
  axes[4].axes.plot(A_Times,AE_Values,'o' , label='AE')
  axes[4].set_xticks(np.linspace(S_DOY,E_DOY,(E_DOY-S_DOY+1)))
  axes[4].legend(loc="upper left", prop={'size': 30})
  axes[4].set_xlabel("Time (DOY)")
except:
  pass

plt.savefig(Station+"Plot.jpg",dpi=120, bbox_inches='tight')
plt.show()
```

The result of the code is presented blow:

![alt text](https://github.com/alikhosravi/BBox/blob/main/docs/Aruc-2019-Torkmanchay.png?raw=true)
