from rdflib import Graph, Literal, BNode, Namespace, RDF, URIRef
from rdflib.namespace import DCAT, RDF, FOAF, DCTERMS, XSD

from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import os
from typing import List, Optional

g = Graph()

g.bind("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
g.bind("dcat", "http://www.w3.org/ns/dcat#")
g.bind("rascat", "http://ffrd.fema.gov/rascat/1.0#")

RASCAT = Namespace("http://ffrd.fema.gov/rascat/1.0#")

kanawha = URIRef("http://ffrd.fema.gov/models/kanawha/")
usgs_gages = URIRef("https://waterdata.usgs.gov/monitoring-location/")

g.bind("kanawha", kanawha)
g.bind("usgs_gages", usgs_gages)

@dataclass
class Mesh2D:
    nominalCellSize: float
    breaklinesMinCellSize: float
    breaklinesMaxCellSize: float
    refinementRegionsMinCellSize: float
    refinementRegionsMaxCellSize: float
    cellCount: int

    def add_bnode(self, g: Graph):
        mesh2d = BNode()
        g.add((mesh2d, RDF.type, RASCAT.Mesh2D))
        g.add((mesh2d, RASCAT.nominalCellSize, Literal(self.nominalCellSize)))
        g.add((mesh2d, RASCAT.breaklinesMinCellSize, Literal(self.breaklinesMinCellSize)))
        g.add((mesh2d, RASCAT.breaklinesMaxCellSize, Literal(self.breaklinesMaxCellSize)))
        g.add((mesh2d, RASCAT.refinementRegionsMinCellSize, Literal(self.refinementRegionsMinCellSize)))
        g.add((mesh2d, RASCAT.refinementRegionsMaxCellSize, Literal(self.refinementRegionsMaxCellSize)))
        g.add((mesh2d, RASCAT.cellCount, Literal(self.cellCount)))
        return mesh2d


@dataclass
class Person:
    name: str
    email: str
    organization: str

    def add_bnode(self, g: Graph):
        person = BNode()
        g.add((person, RDF.type, FOAF.Person))
        g.add((person, FOAF.name, Literal(self.name)))
        g.add((person, FOAF.mbox, Literal(self.email)))
        return person


@dataclass(kw_only=True)
class DcatDataset:
    creators: Optional[List[Person]] = field(default_factory=list)
    title: Optional[str] = None
    description: Optional[str] = None
    modified: Optional[datetime] = None

    def add_dcat_terms(self, g: Graph, uri: URIRef):
        for person in self.creators:
            person_bnode = person.add_bnode(g)
            g.add((uri, DCTERMS.creator, person_bnode))
        g.add((uri, DCTERMS.title, Literal(self.title)))
        g.add((uri, DCTERMS.description, Literal(self.description)))
        g.add((uri, DCTERMS.modified, Literal(self.modified)))


@dataclass
class Streamgage(DcatDataset):
    name: str
    id: str
    link: str
    owner: str

    def uri_ref(self):
        return URIRef(self.id, usgs_gages)

    def add_bnode(self, g: Graph):
        streamgage = BNode()
        g.add((streamgage, RDF.type, RASCAT.Streamgage))
        g.add((streamgage, DCTERMS.title, Literal(self.name)))
        g.add((streamgage, DCTERMS.identifier, Literal(self.id)))
        g.add((streamgage, DCAT.landingPage, Literal(self.link)))
        return streamgage

    def to_rdf(self, g: Graph):
        streamgage = URIRef(self.id, usgs_gages)
        g.add((streamgage, RDF.type, RASCAT.Streamgage))
        g.add((streamgage, DCTERMS.title, Literal(self.name)))
        g.add((streamgage, DCTERMS.identifier, Literal(self.id)))


@dataclass
class Hydrodata(DcatDataset):
    start_datetime: datetime
    end_datetime: datetime
    from_streamgage: Streamgage
    # from_hydroevent


class HydrographType(Enum):
    STAGE = "STAGE"
    FLOW = "FLOW"


@dataclass
class Hydrograph(Hydrodata):
    hydrograph_type: HydrographType = HydrographType.FLOW
    nse: Optional[float] = None
    pbias: Optional[float] = None
    rsr: Optional[float] = None
    r2: Optional[float] = None

    def add_bnode(self, g: Graph):
        hydrograph = BNode()
        g.add((hydrograph, RDF.type, RASCAT.Hydrograph))
        g.add((hydrograph, RASCAT.hydrographType, Literal(self.hydrograph_type.value)))
        g.add((hydrograph, RASCAT.startDatetime, Literal(self.start_datetime)))
        g.add((hydrograph, RASCAT.endDatetime, Literal(self.end_datetime)))
        # streamgage = self.from_streamgage.add_bnode(g)
        g.add((hydrograph, RASCAT.fromStreamgage, self.from_streamgage.uri_ref()))
        if self.nse:
            g.add((hydrograph, RASCAT.nse, Literal(self.nse, datatype=XSD.double)))
        if self.pbias:
            g.add((hydrograph, RASCAT.pbias, Literal(self.pbias, datatype=XSD.double)))
        if self.rsr:
            g.add((hydrograph, RASCAT.rsr, Literal(self.rsr, datatype=XSD.double)))
        if self.r2:
            g.add((hydrograph, RASCAT.r2, Literal(self.r2, datatype=XSD.double)))
        return hydrograph


@dataclass
class RasGeometry(DcatDataset):
    ext: str
    mesh2d: Mesh2D


@dataclass
class RasUnsteadyFlow(DcatDataset):
    ext: str
    calibration_hydrographs: Optional[List[Hydrograph]] = None


@dataclass
class RasPlan(DcatDataset):
    ext: str
    geometry: RasGeometry
    flow: RasUnsteadyFlow


class RasStatus(Enum):
    DRAFT = "DRAFT"
    FINAL = "FINAL"
    PUBLIC_RELEASE = "PUBLIC_RELEASE"


@dataclass
class RasModel(DcatDataset):
    filename: str
    ras_version: str
    status: RasStatus
    geometries: List[RasGeometry]
    flows: List[RasUnsteadyFlow]
    plans: List[RasPlan]

    def basename(self):
        """Return the filename without the extension."""
        return os.path.splitext(self.filename)[0]

    def rasfile(self, ext: str):
        """Return the filename with the given extension."""
        return self.basename() + "." + ext

    def rasfile_uri(self, ext: str, base_uri: str):
        """Return the URI for the given extension."""
        return URIRef(self.rasfile(ext), base_uri)

    def to_rdf(self, g: Graph, base_uri: str):
        """Add the model to the graph."""
        model = URIRef(self.filename, base_uri)
        g.add((model, RDF.type, RASCAT.Model))
        g.add((model, DCTERMS.title, Literal(self.title)))
        g.add((model, DCTERMS.description, Literal(self.description)))
        g.add((model, RASCAT.rasVersion, Literal(self.ras_version)))
        g.add((model, DCTERMS.modified, Literal(self.modified)))
        g.add((model, RASCAT.status, Literal(self.status.value)))
        for person in self.creators:
            person_bnode = person.add_bnode(g)
            g.add((model, DCTERMS.creator, person_bnode))

        for geometry in self.geometries:
            geometry_uri = self.rasfile_uri(geometry.ext, base_uri)
            g.add((model, RASCAT.hasGeometry, geometry_uri))
            g.add((geometry_uri, RDF.type, RASCAT.RasGeometry))
            mesh2d = geometry.mesh2d.add_bnode(g)
            g.add((geometry_uri, RASCAT.hasMesh2D, mesh2d))
            geometry.add_dcat_terms(g, geometry_uri)

        for flow in self.flows:
            flow_uri = self.rasfile_uri(flow.ext, base_uri)
            g.add((model, RASCAT.hasUnsteadyFlow, flow_uri))
            g.add((flow_uri, RDF.type, RASCAT.RasUnsteadyFlow))
            flow.add_dcat_terms(g, flow_uri)

            if flow.calibration_hydrographs:
                for hydrograph in flow.calibration_hydrographs:
                    hyd_bnode = hydrograph.add_bnode(g)
                    g.add((flow_uri, RASCAT.hasCalibrationHydrograph, hyd_bnode))

        for plan in self.plans:
            plan_uri = self.rasfile_uri(plan.ext, base_uri)
            g.add((model, RASCAT.hasPlan, plan_uri))
            g.add((plan_uri, RDF.type, RASCAT.RasPlan))
            g.add((plan_uri, RASCAT.hasGeometry, self.rasfile_uri(plan.geometry.ext, base_uri)))
            g.add((plan_uri, RASCAT.hasUnsteadyFlow, self.rasfile_uri(plan.flow.ext, base_uri)))
            plan.add_dcat_terms(g, plan_uri)


mesh2d = Mesh2D(200, 50, 100, 50, 150, 761035)
g01 = RasGeometry("g01", mesh2d)
g02 = RasGeometry("g02", mesh2d)
g02.title = "ElkMiddle_1996"
g02.description = "This geometry is the same as g01. It was created so that Infiltration is not applied to the January 1996 event since the Excess Precipitation was provided by USACE that takes snowmelt into account."
u02 = RasUnsteadyFlow("u02")
u04 = RasUnsteadyFlow("u04")
u05 = RasUnsteadyFlow("u05")
u06 = RasUnsteadyFlow("u06")
p01 = RasPlan("p01", g01, u02)
p01.title = "Unsteady_Mixed_Nov2003"
p01.description = "November 2003.Calibrated to Queen Shoals Gage"
p03 = RasPlan("p03", g02, u05)
p03.title = "Unsteady_Mixed_Jan1996"
p04 = RasPlan("p04", g01, u06)
p04.title = "Unsteady_Mixed_Jun2016"
p06 = RasPlan("p06", g01, u04)
p06.title = "Unsteady_Mixed_Jan1995"

kevan_leelum = Person("Kevan Leelum", "kevan.leelum@wsp.com", "WSP, Inc.")
britton_wells = Person("Britton Wells", "britton.wells@wsp.com", "WSP, Inc.")
masoud_meshkat = Person("Masoud Meshkat", "masoud.meshkat@wsp.com", "WSP, Inc.")

elk_middle = RasModel(
    filename="ElkMiddle.prj",
    ras_version="6.3.1",
    status=RasStatus.FINAL,
    geometries=[g01, g02],
    flows=[u02, u04, u05, u06],
    plans=[p01, p03, p04],
    title="ElkMiddle",
    description="Elk Middle watershed, Kanawha Basin, WV\nINNOVATION PROJECT #2 - 2D RAS FFRD PILOT\nTechnical Advisement to FY21 IRWA with USACE",
    creators=[kevan_leelum, britton_wells, masoud_meshkat],
    modified=datetime(2022, 10, 19, 19, 55),
)

elk_river_near_frametown = Streamgage(
    name = "Elk River Near Frametown",
    id = "03196600",
    owner = "USGS",
    link = "https://waterdata.usgs.gov/wv/nwis/uv?site_no=03196600",
)
elk_river_near_frametown.to_rdf(g)
elk_river_at_clay = Streamgage(
    name = "Elk River at Clay",
    id = "03196800",
    owner = "USGS",
    link = "https://waterdata.usgs.gov/wv/nwis/uv?site_no=03196800",
)
elk_river_at_clay.to_rdf(g)

hydrograph1 = Hydrograph(
    start_datetime=datetime(2003, 11, 11),
    end_datetime=datetime(2003, 11, 21),
    from_streamgage=elk_river_near_frametown,
    hydrograph_type=HydrographType.STAGE,
    nse=0.31117,
    rsr=0.82996,
    pbias=-0.16572,
    r2=0.7701,
)
hydrograph2 = Hydrograph(
    start_datetime=datetime(2003, 11, 11),
    end_datetime=datetime(2003, 11, 21),
    from_streamgage=elk_river_at_clay,
    hydrograph_type=HydrographType.STAGE,
    nse=0.86802,
    rsr=0.36328,
    pbias=-0.13772,
    r2=0.93277,
)
u02.calibration_hydrographs = [hydrograph1, hydrograph2]


elk_middle.to_rdf(g, base_uri=kanawha)

print(g.serialize(format='turtle'))

with open('./kanawha.ttl', 'w') as out:
    out.write(g.serialize(format='turtle'))
