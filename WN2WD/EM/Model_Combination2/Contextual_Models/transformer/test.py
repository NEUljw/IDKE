# encoding:utf-8
import requests
import rdflib

url = 'https://cbdb.fas.harvard.edu/cbdbc/cbdbedit?@2:582768990:2:::2486:20:21:1@@631704567'
strhtml = requests.get(url)
print(strhtml.text.encode('iso-8859-1').decode('big5'))
