class FFRDMeta:
    def __init__(self):
        self.namespaces = [
            "ffrd_orgs",
            "ffrd_people",
            "kanawha_calibration",
            "kanawha_events",
            "kanawha_misc",
            "kanawha_models",
        ]
        self.vocabularies = ["owl", "rdf", "rdfs", "xsd", "xml", "dcterms", "foaf", "rascat"]
        self.rascat_ttl = "http://raw.githubusercontent.com/arc-pts/ffrd-metadata/main/rascat.ttl#"

    def query_creator(self, model_creator: str) -> str:
        return f"""
        PREFIX rascat: <{self.rascat_ttl}>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT DISTINCT ?title ?description ?model
        WHERE {{
            ?model a rascat:RasModel .
            ?model dcterms:title ?title .
            ?model dcterms:description ?description .
            ?model dcterms:creator ?creators .
            ?model dcterms:creator [foaf:name "{model_creator}"] .
        }}"""

    def query_org(self, org_name: str) -> str:
        return f"""
        PREFIX rascat: <{self.rascat_ttl}>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT DISTINCT ?title ?description ?model ?orgName ?jvName
        WHERE {{
            ?model a rascat:RasModel .
            ?model dcterms:title ?title .
            ?model dcterms:description ?description .
            ?model dcterms:creator ?creator .
            ?creator foaf:member ?org .
            ?org foaf:name ?orgName .
            ?org foaf:member ?jv .
            ?jv foaf:name ?jvName .
            FILTER (?orgName = "{org_name}")
        }}
        """

    def query_cell_count(self, cell_count: str) -> str:
        return f"""
        PREFIX rascat: <{self.rascat_ttl}>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        SELECT ?title ?description ?model ?geometry ?cellCount
        WHERE {{
            ?model a rascat:RasModel .
            ?model dcterms:title ?title .
            ?model dcterms:description ?description .
            ?model rascat:hasGeometry ?geometry .
            ?geometry rascat:hasMesh2D ?mesh2D .
            ?mesh2D rascat:cellCount ?cellCount .
            FILTER (?cellCount > {cell_count})
        }}
        ORDER BY DESC(?cellCount)    
        """

    def query_lulc(self) -> str:
        return f"""
        PREFIX rascat: <{self.rascat_ttl}>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX usgs_gages: <https://waterdata.usgs.gov/monitoring-location/>
        SELECT ?landuseDesc (GROUP_CONCAT(DISTINCT ?title; separator=", ") as ?titles)
        WHERE {{
            ?model a rascat:RasModel .
            ?model dcterms:title ?title .
            ?model dcterms:description ?description .
            ?model rascat:hasGeometry ?geom .
            ?geom rascat:hasRoughness ?rough .
            ?rough rascat:hasLanduseLandcover ?landuse .
            ?landuse dcterms:description ?landuseDesc .
        }}
        GROUP BY ?landuseDesc
        """

    def query_gage(self, gage: str) -> str:
        return f"""
        PREFIX rascat: <{self.rascat_ttl}>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX usgs_gages: <https://waterdata.usgs.gov/monitoring-location/>
        SELECT DISTINCT ?model ?gage ?gageID
        WHERE {{
            ?model a rascat:RasModel .
            ?model rascat:hasPlan ?plan .
            ?plan rascat:hasCalibration ?calib .
            ?calib rascat:fromStreamgage ?gage .
            ?gage dcterms:identifier ?gageID .
            FILTER (?gageID = "{gage}")
        }}
        """

    def query_gages(self, model: str) -> str:
        return f"""
        PREFIX rascat: <{self.rascat_ttl}>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX usgs_gages: <https://waterdata.usgs.gov/monitoring-location/>
        PREFIX kanawha_models: <http://example.ffrd.fema.gov/kanawha/models/>
        SELECT DISTINCT ?model ?gage ?gageID
        WHERE {{
            ?model a rascat:RasModel .
            ?model rascat:hasPlan ?plan .
            ?plan rascat:hasCalibration ?calib .
            ?calib rascat:fromStreamgage ?gage .
            ?gage dcterms:identifier ?gageID .
            FILTER (?model = kanawha_models:{model})
        }}
    """

    def query_calibration(self, limit: str) -> str:
        return f"""
        PREFIX rascat: <{self.rascat_ttl}>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX usgs_gages: <https://waterdata.usgs.gov/monitoring-location/>
        SELECT ?title ?description ?model ?flow ?gage ?nse ?flowTitle ?gageTitle ?hydroType ?plan ?planTitle
        WHERE {{
            ?model a rascat:RasModel .
            ?model dcterms:title ?title .
            ?model dcterms:description ?description .
            ?model rascat:hasPlan ?plan .
            ?plan rascat:hasUnsteadyFlow ?flow .
            ?plan rascat:hasCalibration ?calib .
            ?plan dcterms:title ?planTitle .
            ?flow dcterms:title ?flowTitle .
            ?calib rascat:fromStreamgage ?gage .
            ?calib rascat:hydrographType ?hydroType .
            ?gage dcterms:identifier ?gageID .
            ?gage dcterms:title ?gageTitle .
            ?hydro rascat:nse ?nse .
            FILTER (?hydroType = "Flow")
        }}
        ORDER BY ASC(?nse)
        LIMIT {limit}
        """
