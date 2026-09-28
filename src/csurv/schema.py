import sqlite3

REQUIRED_FEATURES = ["first_reservation_source"]

def create_tables(conn:sqlite3.Connection) -> None:
    """ Create five tables: Customer Features, Customers, Products, Visits, and Purchases"""

    conn.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            first_visit_date TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            visit_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            visit_date TEXT NOT NULL,
            party_size INTEGER NOT NULL,
            reservation_type TEXT NOT NULL,
            special_offer_category TEXT,
            day_of_week TEXT NOT NULL,
            partial_full_discount TEXT,
            discount_reason TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS purchases (
        purchase_id INTEGER PRIMARY KEY,
        visit_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        FOREIGN KEY (visit_id) REFERENCES visits (visit_id),
        FOREIGN KEY (product_id) REFERENCES products (product_id)


        )
    
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS customer_features (
        customer_id INTEGER NOT NULL,
        feature_name TEXT NOT NULL,
        feature_value TEXT NOT NULL,
        PRIMARY KEY (customer_id, feature_name),
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
        )   
    """)

    conn.commit()
