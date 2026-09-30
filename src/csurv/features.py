import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import json
import sqlite3
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder


def readdb(dbpath : str) ->  tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(dbpath)
    customers = pd.read_sql(
        """
        SELECT * 
        FROM CUSTOMERS
        """,
        conn
    )


    visits = pd.read_sql(
        """
        SELECT * 
        FROM VISITS
        """,
        conn
    )


    purchases = pd.read_sql(
        """
        SELECT a.purchase_id, a.visit_id, b.name, b.category, a.quantity
        FROM PURCHASES as a
        LEFT JOIN
        PRODUCTS as b
        ON
        a.product_id=b.product_id
        """,
        conn
    )

    features = pd.read_sql(
        """
        SELECT * 
        FROM customer_features
        """, 
        conn)

    #Transpose Flexible Customer Features
    features_wide = features.pivot(index="customer_id", columns="feature_name", values="feature_value")
    customers_descriptive = customers.merge(features_wide, on="customer_id", how="left")
    visits["visit_date"] = pd.to_datetime(visits["visit_date"])
    
    #Compute Derived Customer Features
    customers_descriptive["first_visit_date"] = pd.to_datetime(customers["first_visit_date"])
    conn.close()
    return customers_descriptive,visits,purchases

def embed_tickets(menu_items : pd.DataFrame,method='proportion') -> pd.DataFrame :
    if method=='proportion':
        tickets_by_cat = (menu_items
                        .groupby(['visit_id','category'])['quantity']
                        .sum()
                        .unstack(fill_value=0)
        )

        ticket_totals = tickets_by_cat.sum(axis='columns')
        category_proportions = tickets_by_cat.div(ticket_totals,axis="rows")
        category_proportions['n_items'] = ticket_totals
        return category_proportions.reset_index()
    

def tte_structure(df : pd.DataFrame,datacutdt : pd.Timestamp) -> pd.DataFrame:
    gb: DataFrameGroupBy = df.groupby('customer_id')

    gaps = []
    for customer, history in gb:
        history = history.sort_values("visit_date").reset_index(drop=True)
        dates = history["visit_date"].tolist()
        temp_gaps=[]
        for i in range(len(dates)):
            avg_gap_days = np.nan
            prev_gap_days = np.nan
            avg_party_size = np.mean([history['party_size'].tolist()[j] for j in range(0,i+1)])
            if i>1:
                avg_gap_days = np.mean(temp_gaps)
                prev_gap_days = temp_gaps[i-1] 
            if i < len(dates) - 1:
                temp_gaps.append((dates[i+1]-dates[i]).days)
                gaps.append(
                    {"customer_id":customer,
                     "visit_id":history["visit_id"].tolist()[i],
                    "gap_num":i,
                    "time":(dates[i+1]-dates[i]).days,
                    "days_since_first_visit":(dates[i+1]-dates[0]).days,
                    "event":1,
                    "avg_gap_days":avg_gap_days,
                    "prev_gap_days":prev_gap_days,
                    "avg_party_size":avg_party_size
                    }
                )
                    
            else:
                gaps.append(
                    {"customer_id":customer,
                     "visit_id":history["visit_id"].tolist()[i],
                    "gap_num":i,
                    "time":(datacutdt - dates[i]).days,
                    "days_since_first_visit":(datacutdt-dates[0]).days,
                    "event":0,
                    "avg_gap_days":avg_gap_days,
                    "prev_gap_days":prev_gap_days,
                    "avg_party_size":avg_party_size
                    }
                )
    return pd.DataFrame(gaps)

def encode_features(df: pd.DataFrame, features: list, methods: dict = {}):
    encoded_parts = []
    cols_to_drop = []
    encoding_metadata = {}  # Save for predict.py

    for feat in features:
        #Get encoding method from the methods dictionary.  Default to onehot
        method = methods.get(feat, "onehot")

        if method == "onehot":
            encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore'
            ).set_output(transform="pandas")

            encoded = encoder.fit_transform(df[[feat]].fillna("None"))

            encoded_parts.append(encoded)
            cols_to_drop.append(feat)

            # Store the categories so we can recreate the encoding
            encoding_metadata[feat] = {
                "method": "onehot",
                "categories": encoder.categories_[0].tolist()
            }

        elif method == "binary":
            df[feat + "_flag"] = df[feat].notna().astype(int)
            cols_to_drop.append(feat)
            encoding_metadata[feat] = {"method": "binary"}

        elif method == "cyclical":
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
            day_num = df[feat].map({d: i for i, d in enumerate(day_order)})
            df[feat + "_sin"] = np.sin(2 * np.pi * day_num / 7)
            df[feat + "_cos"] = np.cos(2 * np.pi * day_num / 7)
            cols_to_drop.append(feat)
            encoding_metadata[feat] = {
                "method": "cyclical",
                "order": day_order
            }

    df = df.drop(columns=cols_to_drop)
    if encoded_parts:
        df = pd.concat([df.reset_index(drop=True)] +
                       [e.reset_index(drop=True) for e in encoded_parts],
                       axis='columns')
    return df, encoding_metadata




def build_features(dbpath : str ="data/demo.db",datacut : str ="2025-12-31") -> pd.DataFrame :
    datacutdt = pd.Timestamp(datacut)
    customers, visits, purchases = readdb(dbpath) 
    ticket_embedding = embed_tickets(purchases)
    tte = tte_structure(visits,datacutdt)
    visits_all = (visits
                  .merge(customers,how='inner',on='customer_id')
                  .merge(ticket_embedding,how="left",on="visit_id")
                  .merge(tte,how='left',on=['visit_id','customer_id'])
    )
    visits_final, encoding_metadata = encode_features(visits_all,
                    features=["first_reservation_source", 
                              'VIP',
                              'Amex Member',
                              "reservation_type",
                              "special_offer_category",
                              "partial_full_discount",
                              "discount_reason",
                              "day_of_week"
                              ],
                    methods={"day_of_week" : "cyclical","discount_reason":"binary","partial_full_discount":"binary"})

    with open("models/encoding_metadata.json", "w") as f:
        json.dump(encoding_metadata, f, indent=2)

    return(visits_final)

if __name__ == "__main__":
    df = build_features()
    print(df.shape)
    print(df.columns.tolist())
    print(df.head())


