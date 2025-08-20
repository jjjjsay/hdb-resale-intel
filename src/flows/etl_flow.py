
from prefect import flow, task
from ..ingest.hdb import fetch_hdb_resale
from ..ingest.mrt import fetch_mrt_exits
from ..ingest.schools import fetch_schools
from ..transform.features import build_features
from ..models.train import train_model

@task
def ingest():
    hdb = fetch_hdb_resale(sample=True)
    mrt = fetch_mrt_exits(sample=True)
    sch = fetch_schools(sample=True)
    return hdb, mrt, sch

@task
def transform(hdb, mrt, sch):
    return build_features(hdb, mrt, sch)

@task
def model(features):
    return train_model(features)

@flow(name="hdb-resale-etl")
def main():
    hdb, mrt, sch = ingest()
    features = transform(hdb, mrt, sch)
    metrics = model(features)
    print({"metrics": metrics, "features_csv": str(features)})

if __name__ == "__main__":
    main()
