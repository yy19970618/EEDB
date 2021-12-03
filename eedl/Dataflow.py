import os
import pickle
import time
import logging
from typing import Dict, Any, Tuple, NamedTuple, Optional

from eedl.constants import DATA_ROOT
from eedl.dtypes import is_categorical
from eedl.dataset.dataset import Table, load_table
from eedl.estimator.eedl.database import DBInterface
from eedl.estimator.eedl.memory import Memory

L = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
L.addHandler(logging.StreamHandler())


class Query(NamedTuple):
    """predicate of each attritbute are conjunctive"""
    predicates: Dict[str, Optional[Tuple[str, Any]]]
    ncols: int


class Label(NamedTuple):
    cardinality: int
    selectivity: float


def getQueryData(data_root: str, workload: str):
    with open(data_root + f"/{workload}/workload/total_query.pkl", "rb") as f:
        return pickle.load(f)


def getTable(dataset, version):
    table = Table(dataset, version)
    return table


def query_2_sql(query: Query, table: Table, aggregate=True, split=False, dbms='postgres'):
    preds = []
    for col, pred in query.predicates.items():
        if pred is None:
            continue
        op, val = pred
        if is_categorical(table.data[col].dtype):
            val = f"\'{val}\'" if not isinstance(val, tuple) else tuple(f"\'{v}\'" for v in val)
        if op == '[]':
            if split:
                preds.append(f"{col} >= {val[0]}")
                preds.append(f"{col} <= {val[1]}")
            else:
                preds.append(f"({col} between {val[0]} and {val[1]})")
        else:
            preds.append(f"{col} {op} {val}")

    if dbms == 'mysql':
        return f"SELECT {'COUNT(*)' if aggregate else '*'} FROM `{table.name}` WHERE {' AND '.join(preds)}"
    return f"SELECT {'COUNT(*)' if aggregate else '*'} FROM \"{table.name}\" WHERE {' AND '.join(preds)}"


if __name__ == "__main__":
    dataset = "power7"
    query_total = getQueryData(DATA_ROOT, dataset)
    table = getTable(dataset, "original")

    db = DBInterface(dataset + "_original")
    initial_query_num = 20000

    data_pool = Memory()
    for index in range(initial_query_num):
        sql = query_2_sql(query_total[index], table).replace("COUNT(*)", "*")
        print(sql)
        card = db.getEsCard(sql)
        data_pool.save_store(query_total[index], card)
