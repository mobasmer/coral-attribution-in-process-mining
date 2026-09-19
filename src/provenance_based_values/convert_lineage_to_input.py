import csv
import json
import os
import time
from datetime import datetime

import psycopg2

from src.provenance_based_values.provenance_queries import QueryBuilder

DB_NAME = "bpi17"
DB_USER = "postgres"
DB_HOST = "localhost"
DB_PORT = 5433

conn = psycopg2.connect(database=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT)
conn.autocommit = True
cur = conn.cursor()

schemata = ["offer", "application", "workflow"]
granularities = ["event", "context", "relation"]
analyses = ["freq_handover", "alternate_response_viol", "efg", "freq_efg"]

TIMEOUT_SECONDS = 600

num_runs = 3

lineage_root = "/Users/maikebasmer/Projects/CORAL-contributions-measurement/data/lineage"
results_root = "/Users/maikebasmer/Projects/CORAL-contributions-measurement/data/results"

timing_records = []
def parse_dnf_lineage(lineage_str):
    """
    Parse DNF lineage strings of the form:
    ((1144654 ⊗ 1144658) ⊕ (1088201 ⊗ 1088205) ⊕ ...)

    Returns a list of lists in JSON-compatible format.
    """
    if not lineage_str or lineage_str.strip() == "":
        return []

    # Remove outermost parentheses
    lineage_str = lineage_str.strip()
    if lineage_str.startswith('(') and lineage_str.endswith(')'):
        lineage_str = lineage_str[1:-1]

    # Split by ⊕ to get OR clauses
    clauses = lineage_str.split('⊕')

    # For each clause, extract numbers
    result = list()
    for clause in clauses:
        # Remove parentheses from clause
        clause = clause.strip().strip('()')
        # Extract all numbers
        vars = clause.split('⊗')
        vars = [var.strip() for var in vars]
        result.append(vars)

    return result

def parse_max_lineage(lineage_str):
    """
    Parse DNF lineage strings of the form:
    (998106 ⊗ 998110) * 1 + (300938 ⊗ 300939) * 1 + (130500 ⊗ 130507) * 1 + (278482 ⊗ 278484) * 1 + ...

    Returns a list of tuples with lists of ids and a value in JSON-compatible format.
    """
    if not lineage_str or lineage_str.strip() == "":
        return []

    lineage_str = lineage_str.strip().strip('max(').rstrip(')')
    # Split by ⊕ to get OR clauses
    components = lineage_str.split(',')

    # For each clause, extract ids
    result = list()
    for component in components:
        parts = component.split('*')
        value = float(parts[1].strip()) if len(parts) > 1 else None
        # Remove parentheses from clause
        clause = parts[0].strip().strip('(').strip(')')
        # Extract all prov ids
        vars = clause.split('⊗')
        vars = [var.strip() for var in vars]
        result.append((vars, value))

    return result

def parse_bnp_lineage(lineage_str):
    """
    Parse DNF lineage strings of the form:
    (998106 ⊗ 998110) * 1 + (300938 ⊗ 300939) * 1 + (130500 ⊗ 130507) * 1 + (278482 ⊗ 278484) * 1 + ...

    Returns a list of tuples with lists of ids and a value in JSON-compatible format.
    """
    if not lineage_str or lineage_str.strip() == "":
        return []

    # Remove outermost parentheses
    lineage_str = lineage_str.strip()

    # Split by ⊕ to get OR clauses
    clauses = lineage_str.split('+')

    # For each clause, extract ids
    result = list()
    for clause in clauses:
        parts = clause.split('*')
        value = int(parts[1].strip()) if len(parts) > 1 else None
        # Remove parentheses from clause
        clause = parts[0].strip().strip('()')
        # Extract all prov ids
        vars = clause.split('⊗')
        vars = [var.strip() for var in vars]
        result.append((vars, value))

    return result

def compute_lineage(query_builder, path, type="simple", timeout_seconds=None):
    print(query_builder.schema, query_builder.granularity, type)
    start_time = time.perf_counter()

    selection = {
        "simple": (query_builder.get_lineage_for_simple_dfg, parse_dnf_lineage),
        "frequency": (query_builder.get_lineage_for_frequency_dfg, parse_bnp_lineage),
        "max": (query_builder.get_lineage_for_max_dfg,parse_max_lineage),
        "efg": (query_builder.get_lineage_for_simple_efg, parse_dnf_lineage),
        "freq_efg": (query_builder.get_lineage_for_freq_efg, parse_bnp_lineage),
        "alternate_response_viol": (query_builder.get_lineage_for_violation_alternate_response, parse_dnf_lineage),
        "alternate_precedence_viol": (query_builder.get_lineage_for_violation_alternate_precedence, parse_dnf_lineage),
        "simple_handover": (query_builder.get_lineage_for_simple_handover, parse_dnf_lineage),
        "freq_handover": (query_builder.get_lineage_for_freq_handover, parse_bnp_lineage)
    }

    query, parser = selection[type]

    if timeout_seconds is not None:
        cur.execute(f"SET statement_timeout = {int(timeout_seconds * 1000)}")

    try:
        cur.execute(query())
        query_result = cur.fetchall()
    except psycopg2.errors.QueryCanceled:
        print(f"Timed out after {timeout_seconds}s computing {type} lineage "
              f"for {query_builder.schema}/{query_builder.granularity}, skipping.")
        duration = time.perf_counter() - start_time
        timing_records.append({
            "step": "compute_lineage",
            "schema": query_builder.schema,
            "granularity": query_builder.granularity,
            "lineage_type": type,
            "duration_seconds": duration,
            "timed_out": True,
        })
        return
    finally:
        if timeout_seconds is not None:
            cur.execute("SET statement_timeout = 0")

    json_provenance = []

    for res in query_result:
        src_activity, dest_activity, frequency, formula, _ = res
        json_provenance.append({
            "src_activity": src_activity,
            "dest_activity": dest_activity,
            "value": frequency,
            "formula": parser(formula),
            "formula_str": formula
        })

    with open(os.path.join(path, f"{type}_lineage.json"), "w") as f:
        json.dump(json_provenance, f, indent=2)

    duration = time.perf_counter() - start_time
    timing_records.append({
        "step": "compute_lineage",
        "schema": query_builder.schema,
        "granularity": query_builder.granularity,
        "lineage_type": type,
        "duration_seconds": duration,
    })

def build_provenance(query_builder, path):
    start_time = time.perf_counter()
    try:
        cur.execute(query_builder.add_provenance())
    except psycopg2.errors.DuplicateColumn:
        cur.execute(query_builder.remove_provenance())

        get_grouping(query_builder, path)

        cur.execute(query_builder.add_provenance())

    print("added provenance")
    cur.execute(query_builder.add_provenance_mapping())
    print("added provenance mapping")

    duration = time.perf_counter() - start_time

    timing_records.append({
        "step": "build_provenance",
        "schema": query_builder.schema,
        "granularity": query_builder.granularity,
        "lineage_type": "",
        "duration_seconds": duration,
    })

def write_timing_results():
    os.makedirs(results_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_root, f"lineage_timing_{timestamp}.csv")

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "schema", "granularity", "lineage_type", "duration_seconds", "timed_out"])
        writer.writeheader()
        writer.writerows(timing_records)

    print(f"Wrote timing results to {results_path}")

def get_grouping(query_builder, path):
    cur.execute(query_builder.get_grouping())
    query_result = cur.fetchall()
    grouping = dict()
    for res in query_result:
        event_idx, c_id = res
        grouping[event_idx] = c_id

    with open(os.path.join(path, f"grouping.json"), "w") as f:
        json.dump(grouping, f, indent=2)

def main():
    for schema in schemata:
        for granularity in granularities:
            for atype in analyses:
                if atype != "freq_handover" and granularity == 'relation':
                    continue

                path = os.path.join(lineage_root, schema, granularity)
                os.makedirs(path, exist_ok=True)
                print(f"Ensured: {path}")

                query_builder = QueryBuilder(schema, granularity)
                cur.execute(query_builder.prepare_db())
                print("prepared db")

                build_provenance(query_builder, path)

                compute_lineage(query_builder, path, type=atype, timeout_seconds=TIMEOUT_SECONDS)

                cur.execute(query_builder.remove_provenance())

    write_timing_results()

if __name__ == "__main__":
    main()

