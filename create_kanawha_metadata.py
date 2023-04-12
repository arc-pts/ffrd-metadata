import dateutil.parser
from rdflib import Graph, Literal, BNode, Namespace, RDF, URIRef
from rdflib.namespace import DCAT, RDF, FOAF, DCTERMS, XSD
import yaml

from datetime import datetime, date, time
from dataclasses import dataclass, field
from enum import Enum
import os
from typing import List, Optional, Union

RASCAT_URI = "http://www.example.org/rascat/0.1#"

g = Graph()

g.bind("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
g.bind("dcat", "http://www.w3.org/ns/dcat#")
g.bind("rascat", RASCAT_URI)

RASCAT = Namespace(RASCAT_URI)

kanawha = URIRef("http://ffrd.fema.gov/models/kanawha/")
usgs_gages = URIRef("https://waterdata.usgs.gov/monitoring-location/")

g.bind("kanawha", kanawha)
g.bind("usgs_gages", usgs_gages)


@dataclass
class Mesh2D:
    nominal_cell_size: float
    breaklines_min_cell_size: Optional[float]
    breaklines_max_cell_size: Optional[float]
    refinement_regions_min_cell_size: Optional[float]
    refinement_regions_max_cell_size: Optional[float]
    cell_count: int

    def add_bnode(self, g: Graph):
        mesh2d = BNode()
        g.add((mesh2d, RDF.type, RASCAT.Mesh2D))
        g.add((mesh2d, RASCAT.nominalCellSize, Literal(self.nominal_cell_size)))
        if self.breaklines_min_cell_size is not None:
            g.add((mesh2d, RASCAT.breaklinesMinCellSize, Literal(self.breaklines_min_cell_size)))
        if self.breaklines_max_cell_size is not None:
            g.add((mesh2d, RASCAT.breaklinesMaxCellSize, Literal(self.breaklines_max_cell_size)))
        if self.refinement_regions_min_cell_size is not None:
            g.add((mesh2d, RASCAT.refinementRegionsMinCellSize, Literal(self.refinement_regions_min_cell_size)))
        if self.refinement_regions_max_cell_size is not None:
            g.add((mesh2d, RASCAT.refinementRegionsMaxCellSize, Literal(self.refinement_regions_max_cell_size)))
        g.add((mesh2d, RASCAT.cellCount, Literal(self.cell_count)))
        return mesh2d


@dataclass
class Person:
    name: str
    email: str
    # organization: str

    def add_bnode(self, g: Graph):
        person = BNode()
        g.add((person, RDF.type, FOAF.Person))
        g.add((person, FOAF.name, Literal(self.name)))
        g.add((person, FOAF.mbox, Literal(self.email)))
        # g.add((person, FOAF.Organization, Literal(self.organization)))
        return person


@dataclass(kw_only=True)
class DcatDataset:
    creators: Optional[List[Person]] = field(default_factory=list)
    title: Optional[str] = None
    description: Optional[str] = None
    modified: Optional[datetime|str] = None
    relation: Optional[str|List[str]] = None

    def __post_init__(self):
        if isinstance(self.modified, str):
            self.modified = dateutil.parser.parse(self.modified)

    def add_dcat_terms(self, g: Graph, uri: URIRef):
        for person in self.creators:
            person_bnode = person.add_bnode(g)
            g.add((uri, DCTERMS.creator, person_bnode))
        if self.title is not None:
            g.add((uri, DCTERMS.title, Literal(self.title)))
        if self.description is not None:
            g.add((uri, DCTERMS.description, Literal(self.description)))
        if self.modified is not None:
            g.add((uri, DCTERMS.modified, Literal(self.modified)))
        if self.relation is not None:
            if isinstance(self.relation, str):
                g.add((uri, DCTERMS.relation, URIRef(self.relation)))
            else:
                for relation in self.relation:
                    g.add((uri, DCTERMS.relation, URIRef(relation)))


@dataclass
class Streamgage(DcatDataset):
    name: str
    id: str
    owner: str = 'USGS'
    uri: Optional[str] = None

    def uri_ref(self):
        if self.owner == 'USGS':
            return URIRef(self.id, usgs_gages)
        return URIRef(self.uri)

    def add_bnode(self, g: Graph):
        streamgage = BNode()
        g.add((streamgage, RDF.type, RASCAT.Streamgage))
        g.add((streamgage, DCTERMS.title, Literal(self.name)))
        g.add((streamgage, DCTERMS.identifier, Literal(self.id)))
        g.add((streamgage, DCTERMS.publisher, Literal(self.owner)))
        return streamgage

    def to_rdf(self, g: Graph):
        if self.owner == 'USGS':
            streamgage = URIRef(self.id, usgs_gages)
        else:
            streamgage = URIRef(self.uri)
        g.add((streamgage, RDF.type, RASCAT.Streamgage))
        g.add((streamgage, DCTERMS.title, Literal(self.name)))
        g.add((streamgage, DCTERMS.identifier, Literal(self.id)))
        g.add((streamgage, DCTERMS.publisher, Literal(self.owner)))


def to_datetime(d: date | str | datetime) -> datetime:
    if isinstance(d, date):
        return datetime.combine(d, time.min)
    if isinstance(d, str):
        return dateutil.parser.parse(d)
    if isinstance(d, datetime):
        return d
    else:
        d_type = type(d)
        raise TypeError(f"d must be a date or string (type: {d_type})")


@dataclass
class Hydrodata(DcatDataset):
    start_datetime: datetime | date | str
    end_datetime: datetime | date | str
    # from_hydroevent

    def __post_init__(self):
        self.start_datetime = to_datetime(self.start_datetime)
        self.end_datetime = to_datetime(self.end_datetime)

    def add_hydrodata_terms(self, g: Graph, uri: URIRef):
        g.add((uri, RASCAT.startDateTime, Literal(self.start_datetime)))
        g.add((uri, RASCAT.endDateTime, Literal(self.end_datetime)))

    def add_bnode(self, g: Graph):
        hydrodata = BNode()
        g.add((hydrodata, RDF.type, RASCAT.Hydrodata))
        g.add((hydrodata, RASCAT.startDateTime, Literal(self.start_datetime)))
        g.add((hydrodata, RASCAT.endDateTime, Literal(self.end_datetime)))
        return hydrodata


class HydrographType(Enum):
    STAGE = "Stage"
    FLOW = "Flow"


@dataclass
class Hydrograph(Hydrodata):
    from_streamgage: Streamgage
    hydrograph_type: HydrographType = HydrographType.FLOW
    nse: Optional[float] = None
    pbias: Optional[float] = None
    rsr: Optional[float] = None
    r2: Optional[float] = None

    def add_bnode(self, g: Graph):
        hydrograph = BNode()
        g.add((hydrograph, RDF.type, RASCAT.Hydrograph))
        g.add((hydrograph, RASCAT.hydrographType, Literal(self.hydrograph_type.value)))
        super().add_hydrodata_terms(g, hydrograph)
        if self.from_streamgage:
            g.add((hydrograph, RASCAT.fromStreamgage, self.from_streamgage.uri_ref()))
        if self.nse:
            g.add((hydrograph, RASCAT.nse, Literal(self.nse, datatype=XSD.double)))
        if self.pbias:
            g.add((hydrograph, RASCAT.pbias, Literal(self.pbias, datatype=XSD.double)))
        if self.rsr:
            g.add((hydrograph, RASCAT.rsr, Literal(self.rsr, datatype=XSD.double)))
        if self.r2:
            g.add((hydrograph, RASCAT.r2, Literal(self.r2, datatype=XSD.double)))
        super().add_dcat_terms(g, hydrograph)
        return hydrograph


@dataclass
class Hyetograph(Hydrodata):
    spatially_varied: bool = True

    def add_bnode(self, g: Graph):
        hyetograph = BNode()
        g.add((hyetograph, RDF.type, RASCAT.Hyetograph))
        g.add((hyetograph, DCTERMS.description, Literal(self.description)))
        g.add((hyetograph, RASCAT.startDateTime, Literal(self.start_datetime)))
        g.add((hyetograph, RASCAT.endDateTime, Literal(self.end_datetime)))
        g.add((hyetograph, RASCAT.spatiallyVaried, Literal(self.spatially_varied)))
        return hyetograph


@dataclass
class DEM(DcatDataset):
    uri: str


@dataclass
class Bathymetry(DcatDataset):
    uri: Optional[str] = None


@dataclass
class TerrainModifications(DcatDataset):
    uri: Optional[str] = None


@dataclass
class Terrain(DcatDataset):
    dem: DEM
    bathymetry: Optional[Bathymetry] = None
    modifications: Optional[TerrainModifications] = None

    def add_bnode(self, g: Graph):
        terrain = BNode()
        g.add((terrain, RDF.type, RASCAT.Terrain))
        g.add((terrain, RASCAT.dem, self.dem.uri))
        if self.bathymetry:
            bathymetry = BNode()
            g.add((bathymetry, RDF.type, RASCAT.Bathymetry))
            if self.bathymetry.uri:
                g.add((terrain, RASCAT.hasBathymetry, self.bathymetry.uri))
            else:
                g.add((terrain, RASCAT.hasBathymetry, bathymetry))
            self.bathymetry.add_dcat_terms(g, bathymetry)
        if self.modifications:
            terrain_modifications = BNode()
            if self.modifications.uri:
                g.add((terrain, RASCAT.hasTerrainModifications, self.modifications.uri))
            else:
                g.add((terrain, RASCAT.hasTerrainModifications, terrain_modifications))
            self.modifications.add_dcat_terms(g, terrain_modifications)
        return terrain


@dataclass
class LanduseLandcover(DcatDataset):
    uri: Optional[str] = None


@dataclass
class Roughness(DcatDataset):
    uri: Optional[str] = None
    landuse: Optional[LanduseLandcover] = None

    def add_bnode(self, g: Graph):
        roughness = BNode()
        g.add((roughness, RDF.type, RASCAT.Roughness))
        if self.uri:
            g.add((roughness, RASCAT.uri, Literal(self.uri)))
        if self.landuse:
            if self.landuse.uri:
                g.add((roughness, RASCAT.hasLanduseLandcover, URIRef(self.landuse.uri)))
            else:
                landuse = BNode()
                g.add((landuse, RDF.type, RASCAT.LanduseLandcover))
                g.add((roughness, RASCAT.hasLanduseLandcover, landuse))
                self.landuse.add_dcat_terms(g, landuse)
        return roughness

@dataclass
class Soils(DcatDataset):
    uri: Optional[str] = None


@dataclass
class PrecipLosses(DcatDataset):
    landuse: Optional[LanduseLandcover] = None
    soils: Optional[Soils] = None

    def add_bnode(self, g: Graph):
        precip_losses = BNode()
        g.add((precip_losses, RDF.type, RASCAT.PrecipLosses))
        if self.landuse:
            if self.landuse.uri:
                g.add((precip_losses, RASCAT.hasLanduseLandcover, URIRef(self.landuse.uri)))
            else:
                landuse = BNode()
                g.add((landuse, RDF.type, RASCAT.LanduseLandcover))
                g.add((precip_losses, RASCAT.hasLanduseLandcover, landuse))
                self.landuse.add_dcat_terms(g, landuse)
        if self.soils:
            if self.soils.uri:
                g.add((precip_losses, RASCAT.hasSoils, URIRef(self.soils.uri)))
            else:
                soils = BNode()
                g.add((soils, RDF.type, RASCAT.Soils))
                g.add((precip_losses, RASCAT.hasSoils, soils))
                self.soils.add_dcat_terms(g, soils)
        return precip_losses


@dataclass
class Structures(DcatDataset):
    uri: Optional[str] = None

    def add_bnode(self, g: Graph):
        structures = BNode()
        g.add((structures, RDF.type, RASCAT.StructureData))
        super().add_dcat_terms(g, structures)
        return structures


@dataclass
class RasGeometry(DcatDataset):
    ext: str
    mesh2d: Mesh2D
    terrain: Terrain
    roughness: Roughness
    precip_losses: Optional[PrecipLosses]
    structures: Optional[Structures]


@dataclass
class RasUnsteadyFlow(DcatDataset):
    ext: str
    hyetograph: Optional[Hyetograph] = None
    calibration_hydrographs: Optional[List[Hydrograph]] = None


@dataclass
class RasPlan(DcatDataset):
    ext: str
    geometry: RasGeometry
    flow: RasUnsteadyFlow
    calibration_hydrographs: Optional[List[Hydrograph]] = None


class RasStatus(Enum):
    DRAFT = "Draft"
    FINAL = "Final"
    PUBLIC_RELEASE = "PublicRelease"


def get_ext(filename: str) -> str:
    return os.path.splitext(filename)[1][1:]

def replace_ext(filename: str, new_ext: str) -> str:
    return os.path.splitext(filename)[0] + f'.{new_ext}'


@dataclass
class Projection(DcatDataset):
    filename: str


@dataclass
class RasModel(DcatDataset):
    filename: str
    ras_version: str
    status: RasStatus
    geometries: List[RasGeometry]
    flows: List[RasUnsteadyFlow]
    plans: List[RasPlan]
    projection: Projection
    vertical_datum: str

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
        g.add((model, RDF.type, RASCAT.RasModel))
        g.add((model, RASCAT.rasVersion, Literal(self.ras_version)))
        g.add((model, RASCAT.status, Literal(self.status.value)))
        projection_ref = URIRef(self.projection.filename, base_uri)
        g.add((projection_ref, RDF.type, RASCAT.Projection))
        self.projection.add_dcat_terms(g, projection_ref)
        g.add((model, RASCAT.projection, projection_ref))
        g.add((model, RASCAT.verticalDatum, Literal(self.vertical_datum)))

        super().add_dcat_terms(g, model)

        for geometry in self.geometries:
            geometry_uri = self.rasfile_uri(geometry.ext, base_uri)
            g.add((model, RASCAT.hasGeometry, geometry_uri))
            g.add((geometry_uri, RDF.type, RASCAT.RasGeometry))
            if geometry.mesh2d is not None:
                mesh2d = geometry.mesh2d.add_bnode(g)
                g.add((geometry_uri, RASCAT.hasMesh2D, mesh2d))
            if geometry.terrain is not None:
                terrain = geometry.terrain.add_bnode(g)
                g.add((geometry_uri, RASCAT.hasTerrain, terrain))
            if geometry.roughness is not None:
                roughness = geometry.roughness.add_bnode(g)
                g.add((geometry_uri, RASCAT.hasRoughness, roughness))
            if geometry.precip_losses is not None:
                precip_losses = geometry.precip_losses.add_bnode(g)
                g.add((geometry_uri, RASCAT.hasPrecipLosses, precip_losses))
            if geometry.structures is not None:
                structures = geometry.structures.add_bnode(g)
                g.add((geometry_uri, RASCAT.hasStructureData, structures))
            geometry.add_dcat_terms(g, geometry_uri)

        for flow in self.flows:
            flow_uri = self.rasfile_uri(flow.ext, base_uri)
            g.add((model, RASCAT.hasUnsteadyFlow, flow_uri))
            g.add((flow_uri, RDF.type, RASCAT.RasUnsteadyFlow))
            flow.add_dcat_terms(g, flow_uri)

            if flow.hyetograph is not None:
                hyeto_bnode = flow.hyetograph.add_bnode(g)
                g.add((flow_uri, RASCAT.hasHyetograph, hyeto_bnode))

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
            if plan.calibration_hydrographs:
                for hydrograph in plan.calibration_hydrographs:
                    hyd_bnode = hydrograph.add_bnode(g)
                    g.add((plan_uri, RASCAT.hasCalibrationHydrograph, hyd_bnode))
            plan.add_dcat_terms(g, plan_uri)

with open('./streamgages.yml', 'r') as streamgages_yml:
    streamgages: List[dict] = yaml.load(streamgages_yml, Loader=yaml.FullLoader)

gages_lookup = {}
for streamgage in streamgages:
    gage = Streamgage(
        name=streamgage.get('title'),
        id=streamgage.get('identifier'),
        owner=streamgage.get('owner', 'USGS'),
        uri=streamgage.get('link'),
    )
    gages_lookup[streamgage['identifier']] = gage
    gage.to_rdf(g)

kanawha_data: List[dict] = []
kanawha_yamls = os.listdir('./kanawha-yaml')
for kanawha_yaml in kanawha_yamls:
    with open(os.path.join('./kanawha-yaml', kanawha_yaml), 'r') as yml:
        kanawha_data.append(yaml.load(yml, Loader=yaml.FullLoader))

proj_albers = Projection(
    'mmc_albers_ft.prj',
    title='Albers Equal Area Conic (feet)',
    description='Modified version of ESRI:102309, provided by USACE. Units are in feet, whereas EPSG:102309 is in meters.',
)

kanawha_dem = DEM(
    URIRef('https://femahq.s3.amazonaws.com/kanawha/tiles/1m'),
    title='Kanawha River Basin DEM',
    description='Digital Elevation Model for the Kanawha River Basin, based on USGS 3DEP data.',
)
g.add((kanawha_dem.uri, RDF.type, RASCAT.DEM))
kanawha_dem.add_dcat_terms(g, kanawha_dem.uri)

nlcd = LanduseLandcover(
    URIRef('https://www.mrlc.gov/data/nlcd-2019-land-cover-conus'),
    title='NLCD 2019',
    description='National Land Cover Database 2019 (CONUS);',
)
g.add((nlcd.uri, RDF.type, RASCAT.LanduseLandcover))
nlcd.add_dcat_terms(g, nlcd.uri)

ssurgo = Soils(
    URIRef('https://www.nrcs.usda.gov/resources/data-and-reports/soil-survey-geographic-database-ssurgo'),
    title='SSURGO',
    description='USDA / NRCS Soil Survey Geographic Database (SSURGO)',
)
g.add((ssurgo.uri, RDF.type, RASCAT.Soils))
ssurgo.add_dcat_terms(g, ssurgo.uri)

for model in kanawha_data:
    model_prj = model['model']

    flows = {}
    for flowfile, f in model['flows'].items():
        hyetograph = f.get('hyetograph')
        if hyetograph is not None:
            hyetograph = Hyetograph(
                start_datetime=hyetograph.get('start_datetime'),
                end_datetime=hyetograph.get('end_datetime'),
                description=hyetograph.get('description'),
                spatially_varied=hyetograph.get('spatially_varied', True),
            )

        calibration_hydrographs = []
        for hydrograph in f.get('hydrographs', []):
            hyd = Hydrograph(
                title=hydrograph.get('title'),
                description=hydrograph.get('description'),
                start_datetime=hydrograph.get('start_datetime'),
                end_datetime=hydrograph.get('end_datetime'),
                from_streamgage=gages_lookup.get(hydrograph.get('from_streamgage')),
                hydrograph_type=HydrographType(hydrograph.get('hydrograph_type', 'Flow')),
                nse=hydrograph.get('nse'),
                rsr=hydrograph.get('rsr'),
                pbias=hydrograph.get('pbias'),
                r2=hydrograph.get('r2'),
            )
            calibration_hydrographs.append(hyd)

        flow = RasUnsteadyFlow(
            ext=get_ext(flowfile),
            title=f.get('title'),
            hyetograph=hyetograph,
            calibration_hydrographs=calibration_hydrographs,
        )
        flows[flowfile] = flow

    geometries = {}
    for geomfile, geom in model['geometries'].items():
        mesh2d = geom.get('mesh2d')
        if mesh2d is not None:
            mesh2d = Mesh2D(
                nominal_cell_size=mesh2d.get('nominal_cell_size'),
                breaklines_min_cell_size=mesh2d.get('breaklines_min_cell_size'),
                breaklines_max_cell_size=mesh2d.get('breaklines_max_cell_size'),
                refinement_regions_min_cell_size=mesh2d.get('refinement_regions_min_cell_size'),
                refinement_regions_max_cell_size=mesh2d.get('refinement_regions_max_cell_size'),
                cell_count=mesh2d.get('cell_count'),
            )

        terrain = geom.get('terrain')
        if terrain is not None:
            bathymetry = terrain.get('bathymetry')
            if bathymetry is not None:
                bathymetry = Bathymetry(
                    title=bathymetry.get('title'),
                    description=bathymetry.get('description'),
                    uri=bathymetry.get('uri'),
                )
            modifications = terrain.get('modifications')
            if modifications is not None:
                modifications = TerrainModifications(
                    title=modifications.get('title'),
                    description=modifications.get('description'),
                    uri=modifications.get('uri'),
                )
            terrain = Terrain(dem=kanawha_dem, bathymetry=bathymetry, modifications=modifications)
        else:
            terrain = Terrain(dem=kanawha_dem)

        roughness = geom.get('roughness')
        if roughness is not None:
            landuse = roughness.get('landuse')
            if landuse == 'nlcd':
                landuse = nlcd
            elif landuse is not None:
                landuse = LanduseLandcover(
                    title=landuse.get('title'),
                    description=landuse.get('description'),
                    uri=landuse.get('uri'),
                )
            roughness = Roughness(
                title=roughness.get('title'),
                description=roughness.get('description'),
                uri=roughness.get('uri'),
                landuse=landuse,
            )

        precip_losses = geom.get('precip_losses')
        if precip_losses is not None:
            landuse = precip_losses.get('landuse')
            if landuse == 'nlcd':
                landuse = nlcd
            elif landuse is not None:
                landuse = LanduseLandcover(
                    title=landuse.get('title'),
                    description=landuse.get('description'),
                    uri=landuse.get('uri'),
                )
            soils = precip_losses.get('soils')
            if soils == 'ssurgo':
                soils = ssurgo
            elif soils is not None:
                soils = Soils(
                    title=soils.get('title'),
                    description=soils.get('description'),
                    uri=soils.get('uri'),
                )
            precip_losses = PrecipLosses(
                title=precip_losses.get('title'),
                description=precip_losses.get('description'),
                landuse=landuse,
                soils=soils,
            )

        structures = geom.get('structures')
        if structures is not None:
            structures = Structures(
                title=structures.get('title'),
                description=structures.get('description'),
                uri=structures.get('uri'),
                relation=structures.get('relation'),
            )

        geometry = RasGeometry(
            ext=get_ext(geomfile),
            title=geom.get('title'),
            description=geom.get('description'),
            mesh2d=mesh2d,
            terrain=terrain,
            roughness=roughness,
            precip_losses=precip_losses,
            structures=structures,
        )
        geometries[geomfile] = geometry

    plans = {}
    for planfile, p in model['plans'].items():
        calibration_hydrographs = []
        for hydrograph in p.get('hydrographs', []):
            hyd = Hydrograph(
                title=hydrograph.get('title'),
                description=hydrograph.get('description'),
                start_datetime=hydrograph.get('start_datetime'),
                end_datetime=hydrograph.get('end_datetime'),
                from_streamgage=gages_lookup.get(hydrograph.get('from_streamgage')),
                hydrograph_type=HydrographType(hydrograph.get('hydrograph_type', 'Flow')),
                nse=hydrograph.get('nse'),
                rsr=hydrograph.get('rsr'),
                pbias=hydrograph.get('pbias'),
                r2=hydrograph.get('r2'),
            )
            calibration_hydrographs.append(hyd)
        plan = RasPlan(
            ext=get_ext(planfile),
            title=p.get('title'),
            description=p.get('description'),
            geometry=geometries[replace_ext(model_prj, p.get('geom'))],
            flow=flows[replace_ext(model_prj, p.get('flow'))],
            calibration_hydrographs=calibration_hydrographs,
        )
        plans[planfile] = plan

    creators = []
    for person in model.get('creators', []):
        person = Person(
            name=person.get('name'),
            email=person.get('email'),
        )
        creators.append(person)

    ras_version = model.get('ras_version', '6.3.1')

    ras_model = RasModel(
        filename=model.get('model'),
        title=model.get('title'),
        description=model.get('description'),
        ras_version=ras_version,
        status=RasStatus.FINAL,
        geometries=geometries.values(),
        flows=flows.values(),
        plans=plans.values(),
        creators=creators,
        modified=model.get('modified'),
        projection=proj_albers,
        vertical_datum='NAVD88',
    )
    ras_model.to_rdf(g, base_uri=kanawha)


print(g.serialize(format='turtle'))
with open('./kanawha.ttl', 'w') as out:
    out.write(g.serialize(format='turtle'))
print('Gages:')
print(gages_lookup.keys(), len(gages_lookup.keys()))


# with open('./kanawha.jsonld', 'w') as out:
#     out.write(g.serialize(format='json-ld', indent=2))
