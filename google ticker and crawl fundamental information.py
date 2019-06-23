from bs4 import BeautifulSoup
import csv
import pandas as pd
import os
import random 
import requests
import pandas as pd

headers_Get = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

def google(q):
    """
        Search ticker by name from Google 
    """

    q = list(q)
    for i in range(len(q)):
        if not (q[i].isalpha()):
            q[i]=' '
    q =''.join(q)

    s = requests.Session()
    q = '+'.join(q.split())
    url = 'https://www.google.com/search?q=' + q + '&ie=utf-8&oe=utf-8'
    print(url)
    r = s.get(url, headers=headers_Get)

    soup = BeautifulSoup(r.text, "html.parser")
    output=' '
    yahoo='https://finance.yahoo.com'
    nasdaq='https://www.nasdaq.com'
    bloomberg='https://www.bloomberg.com'
    for searchWrapper in soup.find_all('div', {'class':'r'}): #this line may change in future based on google's web page structure
        url = searchWrapper.find('a')["href"] 
        url = str(url)
        print(url)
        if(url[0:len(yahoo)]==yahoo):
            if '=' in url.split('/')[-1]:
                return url.split('/')[-1].split('=')[-1]
            return url.split('/')[-2]
        if(url[0:len(nasdaq)]==nasdaq):
            return url.split('/')[-1]
        if(url[0:len(bloomberg)]==bloomberg):
            return url.split('/')[-1].split(':')[0]
    return output


# Below is to use pystock-crawler to get fundamental information
pre = '/root/tmp/'
reportname = 'result.csv'
reportname_1= 'result1.csv'

a = 1061165
b = 1061768

df = pd.read_csv(pre+reportname)
ab = df[(df['CIK']==a)|(df['CIK']==b)]
ticker=[]
st = list(set(ab['Name']))
mp=[]

def pystock_crawler(st):
    """
        Use pystock-crawler to get fundamental information
    """
    for i in range(len(st)):
        print("------------------------------------------")
        print(st[i])
        req=google(st[i]+' ticker symbol')
        res=pre+'result/%d.csv'%(i)
        print( req, res )
        mp.append([req,st[i]])
        print(mp[-1])
        print("------------------------------------------")
        ticker.append(req)
            os.system("pystock-crawler reports %s -o %s -s 20130101 -e 20181231"%(req, res))
        if os.path.getsize(res) <=  0:
            os.system("pystock-crawler reports %s -o %s -s 20130101 -e 20181231"%(req, res))
        
        return mp 

# Save to file
pystock_crawler(st)
df_ =pd.DataFrame()
df_['symble']=[i[0] for i in mp]
df_['Name']=[i[1] for i in mp]
df_.to_csv(pre + reportname_1)
