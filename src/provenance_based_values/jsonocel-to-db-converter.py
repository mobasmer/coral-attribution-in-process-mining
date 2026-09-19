import pm4py
import psycopg2
from sqlalchemy import create_engine, MetaData, Table, inspect

DB_NAME = "bpi17"
DB_USER = "postgres"
DB_HOST = "localhost"
DB_PORT = 5433
OCEL_PATH = "../../data/ocels/BPIC17.jsonocel"

def load_into_separate_schemata(ocel):
    conn = psycopg2.connect(database=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT)
    conn.autocommit = True
    cur = conn.cursor()

    # SQLAlchemy engine for df.to_sql()
    engine = create_engine(f"postgresql+psycopg2://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    try:
        ocel = pm4py.read.read_ocel_json(OCEL_PATH)
        #ocel = pm4py.filter_ocel_events_timestamp(ocel, '2016-01-01 00:00:01', '2016-06-30 23:59:59')

        object_types = pm4py.ocel.ocel_get_object_types(ocel)

        for object_type in object_types:
            print(f"Processing object type: {object_type}")

            # Use a safe schema name (replace special chars with underscores)
            schema = object_type.replace(" ", "_").replace("-", "_").replace(":", "_").lower()

            # 1. Create schema
            cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')

            df = pm4py.ocel.ocel_flattening(ocel, object_type=object_type)
            variants = pm4py.stats.get_variants(df)
            print(len(variants))

            # 2. Context table — unique case IDs
            df_context = df[['case:concept:name']].drop_duplicates().rename(
                columns={'case:concept:name': 'context_idx'}
            )
            df_context.to_sql('context', engine, schema=schema, if_exists='replace', index=False)

            # 3. Event table — all events with their attributes
            df_event = df.rename(columns={
                'case:concept:name': 'c_id',
                'concept:name': 'activity',
                'time:timestamp': 'timestamp',
                'ocel:eid': 'event_idx',
                'resource': 'resource_idx'
            })
            df_event.to_sql('event', engine, schema=schema, if_exists='replace', index=False)

            cur.execute(f'CREATE INDEX IF NOT EXISTS idx_event_c_id_timestamp ON "{schema}".event (c_id, timestamp);')

            # 4. Relation table — consecutive event pairs with duration
            df_sorted = df.sort_values(by=['case:concept:name', 'time:timestamp'])
            df_sorted = df_sorted.copy()
            df_sorted['next_event'] = df_sorted.groupby('case:concept:name')['ocel:eid'].shift(-1)
            df_sorted['next_event_timestamp'] = df_sorted.groupby('case:concept:name')['time:timestamp'].shift(-1)
            df_sorted = df_sorted.dropna(subset=['next_event'])
            df_sorted['duration'] = (
                    (df_sorted['next_event_timestamp'] - df_sorted['time:timestamp'])
                    .dt.total_seconds() / 60
            ).round(2)  # Duration in minutes

            df_relation = df_sorted[['case:concept:name', 'ocel:eid', 'next_event', 'duration']].rename(columns={
                'case:concept:name': 'c_id',
                'ocel:eid': 'src_event_id',
                'next_event': 'dest_event_id',
            })

            df_relation.insert(0, 'relation_idx', range(1, len(df_relation) + 1))
            df_relation.to_sql('relation', engine, schema=schema, if_exists='replace', index=False)

            cur.execute(f'CREATE INDEX IF NOT EXISTS idx_relation_src_event_id ON "{schema}".relation (src_event_id);')
            cur.execute(f'CREATE INDEX IF NOT EXISTS idx_relation_dest_event_id ON "{schema}".relation (dest_event_id);')

            print(f"  -> Schema '{schema}' created with tables: context, event, relation")
    finally:
        print("Cleaning up")
        cur.close()
        conn.close()
        engine.dispose()

def load_into_shared_schema(ocel):
    conn = psycopg2.connect(database=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT)
    conn.autocommit = True
    cur = conn.cursor()

    # SQLAlchemy engine for df.to_sql()
    engine = create_engine(f"postgresql+psycopg2://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    try:
        ocel = pm4py.read.read_ocel_json(OCEL_PATH)

        # 1. Create schema
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS ocel_data;')

        object_types = pm4py.ocel.ocel_get_object_types(ocel)
        for object_type in object_types:
            print(f"Processing object type: {object_type}")

            # Use a safe schema name (replace special chars with underscores)
            schema = object_type.replace(" ", "_").replace("-", "_").replace(":", "_").lower()



            df = pm4py.ocel.ocel_flattening(ocel, object_type=object_type)
            variants = pm4py.stats.get_variants(df)
            print(len(variants))

            # 2. Context table — unique case IDs
            df_context = df[['case:concept:name']].drop_duplicates().rename(
                columns={'case:concept:name': 'context_idx'}
            )
            df_context.to_sql('context', engine, schema=schema, if_exists='replace', index=False)

            # 3. Event table — all events with their attributes
            df_event = df.rename(columns={
                'case:concept:name': 'c_id',
                'concept:name': 'activity',
                'time:timestamp': 'timestamp',
                'ocel:eid': 'event_idx',
                'resource': 'resouce_idx'
            })
            df_event.to_sql('event', engine, schema=schema, if_exists='replace', index=False)

            # 4. Relation table — consecutive event pairs with duration
            df_sorted = df.sort_values(by=['case:concept:name', 'time:timestamp'])
            df_sorted = df_sorted.copy()
            df_sorted['next_event'] = df_sorted.groupby('case:concept:name')['ocel:eid'].shift(-1)
            df_sorted['next_event_timestamp'] = df_sorted.groupby('case:concept:name')['time:timestamp'].shift(-1)
            df_sorted = df_sorted.dropna(subset=['next_event'])
            df_sorted['duration'] = (
                    (df_sorted['next_event_timestamp'] - df_sorted['time:timestamp'])
                    .dt.total_seconds() / 60
            ).round(2)  # Duration in minutes

            df_relation = df_sorted[['case:concept:name', 'ocel:eid', 'next_event', 'duration']].rename(columns={
                'case:concept:name': 'c_id',
                'ocel:eid': 'src_event_id',
                'next_event': 'dest_event_id',
            })

            df_relation.insert(0, 'relation_idx', range(1, len(df_relation) + 1))
            df_relation.to_sql('relation', engine, schema=schema, if_exists='replace', index=False)

            print(f"  -> Schema '{schema}' created with tables: context, event, relation")
    finally:
        print("Cleaning up")
        cur.close()
        conn.close()
        engine.dispose()

def load_from_sqlite():
    SQLITE_PATH = "sqlite:///../../data/ocels/BPIC17.sqlite"
    DB_NAME = "bpi17ocel"

    sqlite_engine = create_engine(SQLITE_PATH)
    pg_engine = create_engine(f"postgresql+psycopg2://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    sqlite_meta = MetaData()
    sqlite_meta.reflect(bind=sqlite_engine)

    inspector = inspect(sqlite_engine)
    table_names = inspector.get_table_names()
    print(f"Found {len(table_names)} tables: {table_names}")

    for table_name in table_names:
        print(f"Migrating table: {table_name}")

        sqlite_table = Table(table_name, sqlite_meta, autoload_with=sqlite_engine)

        # Create corresponding table in Postgres
        pg_meta = MetaData()
        pg_table = sqlite_table.to_metadata(pg_meta)
        pg_meta.create_all(pg_engine)

        # Migrate data in batches
        batch_size = 5000
        offset = 0

        with sqlite_engine.connect() as sqlite_conn, pg_engine.begin() as pg_conn:
            while True:
                rows = sqlite_conn.execute(
                    sqlite_table.select().limit(batch_size).offset(offset)
                ).fetchall()

                if not rows:
                    break

                pg_conn.execute(
                    pg_table.insert(),
                    [row._mapping for row in rows]
                )

                offset += batch_size
                print(f"  inserted {offset} rows into {table_name}")

    print("Migration complete")
    sqlite_engine.dispose()
    pg_engine.dispose()


if __name__ == "__main__":
    #load_from_sqlite()
    load_into_separate_schemata(OCEL_PATH)