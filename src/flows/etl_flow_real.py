from prefect import flow, task
from ..ingest_real.hdb_real import fetch_hdb_resale_real
from ..ingest_real.mrt_real import fetch_mrt_exits_real
from ..ingest_real.schools_real import fetch_schools_real
from ..transform.features import build_features
from ..models.train import train_model

@task(retries=2, retry_delay_seconds=10)
def ingest(api_key: str | None = None):
    hdb = fetch_hdb_resale_real(api_key=api_key)
    mrt = fetch_mrt_exits_real(api_key=api_key)
    sch = fetch_schools_real(api_key=api_key)
    return hdb, mrt, sch

@task
def transform(hdb, mrt, sch):
    return build_features(hdb, mrt, sch)

@task
def model(features):
    return train_model(features)

@flow(name="hdb-resale-etl-real")
def main(api_key: str | None = None):
    hdb, mrt, sch = ingest(api_key)
    features = transform(hdb, mrt, sch)
    metrics = model(features)
    print({"metrics": metrics, "features_csv": str(features)})

if __name__ == "__main__":
    main()
