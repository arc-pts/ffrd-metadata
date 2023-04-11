import rdflib

g = rdflib.Graph()
g.parse("./kanawha.ttl", format="turtle")

test_query = """
PREFIX rascat: <http://www.example.org/rascat/0.1#>
PREFIX dcterms: <http://purl.org/dc/terms/>
SELECT ?title ?description ?model
WHERE {
    ?model a rascat:RasModel .
    ?model dcterms:title ?title .
    ?model dcterms:description ?description .
    ?model dcterms:creator ?creators .
    ?model dcterms:creator [foaf:name "Mark McBroom"] .
}
GROUP BY ?title ?description ?model
"""
print(test_query)
qres = g.query(test_query)
for row in qres:
    # print(row)
    title = row[0]
    description = row[1]
    model = row[2]
    print('------------------')
    print(f'MODEL: {model}')
    print(f'TITLE: {title}')
    print(f'DESCRIPTION: {description}')
    print('\n')

test_query2 = """
PREFIX rascat: <http://www.example.org/rascat/0.1#>
PREFIX dcterms: <http://purl.org/dc/terms/>
SELECT ?title ?description ?model ?geometry ?cellCount
WHERE {
    ?model a rascat:RasModel .
    ?model dcterms:title ?title .
    ?model dcterms:description ?description .
    ?model rascat:hasGeometry ?geometry .
    ?geometry rascat:hasMesh2D ?mesh2D .
    ?mesh2D rascat:cellCount ?cellCount .
    FILTER (?cellCount > 400000)
}
"""
print(test_query2)
qres = g.query(test_query2)
for row in qres:
    # print(row)
    title = row[0]
    description = row[1]
    model = row[2]
    geometry = row[3]
    cell_count = row[4]
    print('------------------')
    print(f'MODEL: {model}')
    print(f'TITLE: {title}')
    print(f'DESCRIPTION: {description}')
    print(f'GEOMETRY: {geometry}')
    print(f'CELL COUNT: {cell_count}')
    print('\n')

test_query3 = """
PREFIX rascat: <http://www.example.org/rascat/0.1#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX usgs_gages: <https://waterdata.usgs.gov/monitoring-location/>
SELECT ?title ?description ?model ?flow ?gage ?nse ?flowTitle ?gageTitle ?hydroType ?plan ?planTitle
WHERE {
    ?model a rascat:RasModel .
    ?model dcterms:title ?title .
    ?model dcterms:description ?description .
    ?model rascat:hasPlan ?plan .
    ?plan rascat:hasUnsteadyFlow ?flow .
    ?plan rascat:hasCalibrationHydrograph ?hydro .
    ?plan dcterms:title ?planTitle .
    ?flow dcterms:title ?flowTitle .
    ?hydro rascat:fromStreamgage ?gage .
    ?hydro rascat:hydrographType ?hydroType .
    ?gage dcterms:identifier ?gageID .
    ?gage dcterms:title ?gageTitle .
    ?hydro rascat:nse ?nse .
    FILTER (?nse < 0.0) 
    FILTER (?hydroType = "Flow")
}
"""
print(test_query3)
qres = g.query(test_query3)
for row in qres:
    # print(row)
    title = row[0]
    description = row[1]
    model = row[2]
    flow = row[3]
    gage = row[4]
    nse = row[5]
    flow_title = row[6]
    gage_title = row[7]
    hydro_type = row[8]
    plan = row[9]
    plan_title = row[10]
    print('------------------')
    print(f'MODEL: {model}')
    print(f'TITLE: {title}')
    print(f'DESCRIPTION: {description}')
    print(f'PLAN: {plan_title} ({plan})')
    print(f'FLOW FILE: {flow_title} ({flow})')
    print(f'GAGE: {gage_title} ({gage})')
    print(f'TYPE: {hydro_type}')
    print(f'NSE: {nse}')
    print('\n')