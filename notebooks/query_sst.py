import rdflib
from sparql import FFRDMeta
from datetime import datetime


ffrd = FFRDMeta()

g_kanawha = rdflib.Graph()
g_kanawha.parse("../kanawha/kanawha.ttl", format="turtle")


results = g_kanawha.query(ffrd.query_hydroevents())
for row in results:
    print(row)


# g_aorc = rdflib.Graph()
# g_aorc.parse("https://ckan.dewberryanalytics.com/dataset/9b08c661-3b9c-441f-a4a0-6e1a9c865475/resource/16212ad7-32f9-49c4-b78e-d136bc1425d5/download/2016.jsonld", format="json-ld")




dt1 = datetime(2016, 1, 10, 0, 0, 0)
dt2 = datetime(2016, 1, 30, 0, 0, 0)


results = g_aorc.query(ffrd.query_sst(dt1, dt2))

valid = []
for row in results:
    dist = row[0]
    url = row[1]
    temporal = row[2]
    temporal_dt = datetime.fromisoformat(temporal.split("/")[0])
    if temporal_dt >= dt1 and temporal_dt <= dt2:
        valid.append((dist, url, temporal_dt))
        print(dist, url, temporal_dt)

