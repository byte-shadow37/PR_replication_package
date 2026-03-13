import sqlite3
import os
from pathlib import Path


def merge_db_files(source_folder, output_db='merged_database.db'):
    conn_out = sqlite3.connect(output_db)
    cursor_out = conn_out.cursor()

    output_path = Path(output_db).resolve()
    db_files = [f for f in Path(source_folder).glob('*.db')
                if f.resolve() != output_path]

    if not db_files:
        print(f"No .db files found in {source_folder}")
        conn_out.close()
        return

    print(f"Found {len(db_files)} database files")

    # Iterate over each database file
    for db_file in db_files:
        print(f"\nProcessing: {db_file.name}")

        # Use file name (without extension) as table name prefix
        prefix = db_file.stem.replace('-', '_').replace(' ', '_')

        try:
            # Connect to the source database
            conn_src = sqlite3.connect(str(db_file))
            cursor_src = conn_src.cursor()

            # Get all tables from the source database
            cursor_src.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor_src.fetchall()

            # Copy each table
            for (table_name,) in tables:
                if table_name == 'sqlite_sequence':
                    continue

                # Create new table name (prefix_originalTableName)
                new_table_name = f"{prefix}_{table_name}"

                print(f"  - Copying table: {table_name} -> {new_table_name}")

                # Retrieve table structure and column info
                cursor_src.execute(f"PRAGMA table_info({table_name})")
                columns = cursor_src.fetchall()

                # Build CREATE TABLE statement for the output DB
                col_defs = []
                for col in columns:
                    col_id, col_name, col_type, not_null, default_val, pk = col
                    col_def = f"{col_name} {col_type}"
                    if not_null:
                        col_def += " NOT NULL"
                    if default_val is not None:
                        col_def += f" DEFAULT {default_val}"
                    if pk:
                        col_def += " PRIMARY KEY"
                    col_defs.append(col_def)

                create_sql = f"CREATE TABLE IF NOT EXISTS {new_table_name} ({', '.join(col_defs)})"

                # Create table in the output database
                cursor_out.execute(create_sql)

                # Copy table data
                cursor_src.execute(f"SELECT * FROM {table_name}")
                rows = cursor_src.fetchall()

                if rows:
                    # Get number of columns
                    num_cols = len(rows[0])
                    placeholders = ','.join(['?' for _ in range(num_cols)])

                    cursor_out.executemany(
                        f"INSERT INTO {new_table_name} VALUES ({placeholders})",
                        rows
                    )
                    print(f"    Inserted {len(rows)} rows")
                else:
                    print(f"    Table is empty, no data inserted")

            conn_src.close()

        except Exception as e:
            print(f"Error processing {db_file.name}: {e}")
            continue

    # Commit changes and close the connection
    conn_out.commit()
    conn_out.close()

    print(f"\n✓ Merge completed! Output file: {output_db}")
    print(f"✓ Database size: {os.path.getsize(output_db) / 1024:.2f} KB")


if __name__ == "__main__":
    # Example usage
    source_folder = "/database"  # Change to your folder path
    output_database = "merged_database.db"  # Change to your desired output file name

    merge_db_files(source_folder, output_database)