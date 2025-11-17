import os
import psycopg

dsn=os.environ['PROTOTYPE_DATABASE_URL']
with psycopg.connect(dsn) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                table_name,
                json_agg(json_build_object(
                    'column_name', column_name,
                    'data_type', data_type,
                    'is_nullable', is_nullable,
                    'udt_name', udt_name
                ) ORDER BY ordinal_position) AS columns
            FROM information_schema.columns
            WHERE table_schema = 'public'
            GROUP BY table_name
            ORDER BY table_name;
        """)
        for table_name, columns in cur.fetchall():
            print(f"Table: {table_name}")
            for col in columns:
                print(f"  - {col['column_name']}: {col['data_type']} ({col['udt_name']}) nullable={col['is_nullable']}")
            print()
