import asyncio
import csv
from pathlib import Path

from src.database import get_async_session_context, init_db
from src.models import Query
from src.postgres_repo import PostgresRepo


async def read_csv_and_insert_to_postgres(
    csv_file_path: str | Path, batch_size: int = 1_000_000
) -> None:
    """
    Read CSV file and insert data to PostgreSQL in batches.

    Args:
        csv_file_path: Path to the CSV file
        batch_size: Number of records to process in each batch
    """
    csv_path = Path(csv_file_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Reading CSV file: {csv_path}")
    print(f"Processing in batches of {batch_size}")

    # Initialize database tables
    await init_db()

    total_processed = 0

    async with get_async_session_context() as session:
        postgres_repo = PostgresRepo(session)

        # Read CSV file in chunks to handle large files
        with open(csv_path, "r", encoding="utf-8") as file:
            csv_reader = csv.DictReader(file)

            batch = []
            for row in csv_reader:
                # Create Query object from CSV row
                query_obj = Query(
                    query=str(row["query"].strip()),
                    has_bge_embedding=False,
                    has_xml_embedding=False,
                )
                batch.append(query_obj)

                # Process batch when it reaches the specified size
                if len(batch) >= batch_size:
                    await postgres_repo.insert_queries(batch)
                    total_processed += len(batch)
                    print(f"Processed {total_processed} records...")
                    batch = []

            # Process remaining records in the final batch
            if batch:
                await postgres_repo.insert_queries(batch)
                total_processed += len(batch)
                print(f"Processed final batch. Total records: {total_processed}")

    print(f"✅ Successfully imported {total_processed} queries to PostgreSQL")


async def main():
    """Main function to run the CSV import"""
    # Get the CSV file path relative to the project root
    project_root = Path(__file__).parent.parent
    csv_file = project_root / "data" / "basalam_normalized_queries_20250916.csv"

    try:
        await read_csv_and_insert_to_postgres(csv_file, batch_size=10_000)
    except Exception as e:
        print(f"❌ Error importing CSV: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
