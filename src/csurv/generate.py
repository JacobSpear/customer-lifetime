"""Synthetic data generator for csurv.

Simulates restaurant customer visit data with a Beta-based churn model.
All distributional assumptions are marked with # ASSUMPTION comments.
"""

import sqlite3
import numpy as np
from datetime import date, timedelta
from collections import defaultdict, Counter
from pathlib import Path
import json

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

SIM_START = date(2024, 1, 1)
SIM_END = date(2025, 12, 31)
MENU_SWAP_DATE = date(2025, 1, 1)

SEASONAL_RATES = {
    "winter":  45,   # Jan-Mar
    "spring":  55,   # Apr-May
    "summer":  70,   # Jun-Sep
    "fall":    55,   # Oct-pre-Thanksgiving
    "holiday": 65,   # Thanksgiving-New Year's
}


DOW_PROBS = np.array([0.11, 0.11, 0.12, 0.12, 0.2, 0.18, 0.16])
DOW_PROBS = DOW_PROBS / DOW_PROBS.sum()

PARTY_SIZE_PROBS = np.array([0.05, 0.35, 0.15, 0.3, 0.08, 0.05, 0.02])
PARTY_SIZE_PROBS = PARTY_SIZE_PROBS / PARTY_SIZE_PROBS.sum()
EVENT_SIZE_PROBS = np.array([0.2, 0.2,0.2,0.2,0.2])
EVENT_SIZE_PROBS = EVENT_SIZE_PROBS / EVENT_SIZE_PROBS.sum()

# Weibull return-time
#This models a slow build-up over time of "desire" to return in customers who haven't churned
WEIBULL_SHAPE = 1.5
#Weibull RVs scale linearly under multiplication of the scale parameter
WEIBULL_BASE_SCALE = 2.0  # weeks

# ─────────────────────────────────────────────────────────────────────
# MENU: (name, category)
# ─────────────────────────────────────────────────────────────────────

PRODUCTS = [
    ("Tuna Kinilaw", "appetizer"),          
    ("Bahay Kubo Salad", "appetizer"),   
    ("Sinuglaw", "appetizer"),  
    ("Sinigang na Isda", "entree"),
    ("Adobong Kabute", "entree"),     
    ("Pinakbet", "entree"), 
    ("Humba", "entree"),       
    ("Kaldereta", "entree"),   
    ("Fruit Salad", "dessert"),      
    ("Sapin Sapin", "dessert"),            
    ("Tamarind Sorbet", "dessert"),         
]

APP_IDX = [0, 1, 2]
ENTREE_IDX_BEFORE = [3, 4, 5, 6]
ENTREE_IDX_AFTER  = [3, 4, 5, 7]
DESSERT_IDX = [8, 9, 10]

# ─────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────

CUSTOMER_FEATURES = {
    "first_reservation_source": {
        "values": ["ReservationApp", "Walk-In", "Website","Friend"],
        "dirichlet_alpha": [4, 2, 1, 0.5], 
    },
    "VIP": {
        "values": [None, "Noteworthy Individual", "Friends and Family", "Industry", "Critic/Influencer"],
        "dirichlet_alpha": [20, 1, 2, 1, 0.5], 
    },
    "Amex Member": {
        "values": [None, "Yes"],
        "dirichlet_alpha": [10, 3], 
    },
}

VISIT_FEATURES = {
    "reservation_type": {
        "values": ["Pre Fixe", "A La Carte", "Private Event", "Bar"],
        "dirichlet_alpha": [3, 4, 1, 2],
    },
    "special_offer_random": {
        "values": [None, "Birthday", "Anniversary"],
        "dirichlet_alpha": [15, 1, 1],  
    },
    "discount": {
        "values": [None, "partial+complementary", "partial+complaint", "full+complementary", "full+complaint"],
        "dirichlet_alpha": [30, 2, 1, 1, 0.5],  
    },
}

# ─────────────────────────────────────────────────────────────────────
# BETA ADJUSTMENTS: (scale_a, a1, a2, scale_b, b1, b2)
#   delta_a ~ scale_a * Beta(a1, a2)   (pushes churn UP)
#   delta_b ~ scale_b * Beta(b1, b2)   (pushes churn DOWN)
# ─────────────────────────────────────────────────────────────────────
IMPORTANCE = {'app': 0.5, 'entree': 1.0, 'dessert': 0.7}

BETA_ADJ = {
    # ── Customer features ──────────────────────────────────────────
    # Friend referrals → strong retention
    ("first_reservation_source", "ReservationApp"):  (0.2, 1, 1,  1.0, 1, 3),
    ("first_reservation_source", "Walk-In"):         (0.5, 1, 1,  0.5, 1, 1),   # neutral
    ("first_reservation_source", "Website"):         (0.3, 1, 1,  0.8, 1, 2),   # slight retention
    ("first_reservation_source", "Friend"):          (0.1, 1, 1,  1.8, 1, 4),   # strong retention

    # VIP: Industry & F&F → loyal; Critic → risky
    ("VIP", None):                                   (0.0, 1, 1,  0.0, 1, 1),
    ("VIP", "Noteworthy Individual"):                (0.1, 1, 1,  1.2, 1, 3),
    ("VIP", "Friends and Family"):                   (0.05,1, 1,  1.5, 1, 4),
    ("VIP", "Industry"):                             (0.05,1, 1,  2.0, 1, 5),   # very loyal
    ("VIP", "Critic/Influencer"):                    (1.5, 1, 2,  0.1, 1, 1),   # high churn risk

    # Amex → mild retention
    ("Amex Member", None):                           (0.0, 1, 1,  0.0, 1, 1),
    ("Amex Member", "Yes"):                          (0.1, 1, 1,  0.8, 1, 3),

    # ── Visit features ─────────────────────────────────────────────
    # Pre Fixe → committed diners, strong retention
    ("reservation_type", "Pre Fixe"):                (0.1, 1, 1,  1.2, 1, 3),
    ("reservation_type", "A La Carte"):              (0.3, 1, 1,  0.3, 1, 1),   # neutral
    ("reservation_type", "Private Event"):           (1.2, 1, 2,  0.1, 1, 1),   # high churn (one-off events)
    ("reservation_type", "Bar"):                     (0.6, 1, 1,  0.4, 1, 2),   # slight churn

    # Special offers → mostly retention (people celebrate → positive association)
    ("special_offer", None):                         (0.0, 1, 1,  0.0, 1, 1),
    ("special_offer", "Birthday"):                   (0.1, 1, 1,  1.0, 1, 2),
    ("special_offer", "Anniversary"):                (0.1, 1, 1,  1.2, 1, 3),
    ("special_offer", "Valentines Day"):             (0.8, 1, 2,  0.2, 1, 1),   # high churn (forced occasion)
    ("special_offer", "New Years"):                  (0.7, 1, 2,  0.2, 1, 1),   # high churn (forced occasion)

    # Discounts: comps → retention; complaints → churn
    ("discount", None):                              (0.0, 1, 1,  0.0, 1, 1),
    ("discount", "partial+complementary"):           (0.1, 1, 1,  0.8, 1, 3),
    ("discount", "partial+complaint"):               (1.5, 1, 3,  0.1, 1, 1),   # strong churn
    ("discount", "full+complementary"):              (0.05,1, 1,  1.2, 1, 4),   # strong retention
    ("discount", "full+complaint"):                  (2.0, 1, 4,  0.05,1, 1),   # very strong churn

    # ── Dishes ──────────────────────────────────────────────────────
    # Stars of the menu (retention): Sinigang, Sinuglaw, Tamarind Sorbet
    # Weak dishes (churn): Humba, Fruit Salad, Bahay Kubo Salad
    # Neutral: the rest
    ("dish", "Tuna Kinilaw"):      (IMPORTANCE['app']*0.3,  1, 1,  IMPORTANCE['app']*1.0,  1, 3),    # mild retention
    ("dish", "Bahay Kubo Salad"):  (IMPORTANCE['app']*1.0,  1, 2,  IMPORTANCE['app']*0.2,  1, 1),    # churn (weak dish)
    ("dish", "Sinuglaw"):          (IMPORTANCE['app']*0.1,  1, 1,  IMPORTANCE['app']*1.2,  1, 3),    # star app → retention

    ("dish", "Sinigang na Isda"):  (IMPORTANCE['entree']*0.1, 1, 1, IMPORTANCE['entree']*1.5, 1, 4), # star entree → strong retention
    ("dish", "Adobong Kabute"):    (IMPORTANCE['entree']*0.4, 1, 1, IMPORTANCE['entree']*0.4, 1, 1), # neutral
    ("dish", "Pinakbet"):          (IMPORTANCE['entree']*0.3, 1, 1, IMPORTANCE['entree']*0.8, 1, 3), # mild retention
    ("dish", "Humba"):             (IMPORTANCE['entree']*1.2, 1, 4, IMPORTANCE['entree']*0.1, 1, 1), # weak entree → churn
    ("dish", "Kaldereta"):         (IMPORTANCE['entree']*0.3, 1, 1, IMPORTANCE['entree']*0.6, 1, 2), # mild retention

    ("dish", "Fruit Salad"):       (IMPORTANCE['dessert']*1.0, 1, 3, IMPORTANCE['dessert']*0.1, 1, 1), # weak dessert → churn
    ("dish", "Sapin Sapin"):       (IMPORTANCE['dessert']*0.3, 1, 1, IMPORTANCE['dessert']*0.5, 1, 2), # mild retention
    ("dish", "Tamarind Sorbet"):   (IMPORTANCE['dessert']*0.1, 1, 1, IMPORTANCE['dessert']*1.3, 1, 4), # star dessert → retention
}

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def get_season(d):
    m = d.month
    if (m == 11 and d.day >= 20) or m == 12:
        return "holiday"
    if m in (1, 2, 3):
        return "winter"
    if m in (4, 5):
        return "spring"
    if m in (6, 7, 8, 9):
        return "summer"
    return "fall"

def sample_category(probs, values, rng):
    return values[rng.choice(len(values), p=probs)]

def draw_adjustment(key, rng):
    sa, a1, a2, sb, b1, b2 = BETA_ADJ[key]
    return sa * rng.beta(a1, a2), sb * rng.beta(b1, b2)

def is_special_day(d,desc):
    if desc=="Valentines":
        return d.month == 2 and 13 <= d.day <= 15
    if desc=="New Years Eve":
        return d.month == 12 and d.day == 31




# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def generate_data(db_path, seed=42):

    Path(db_path).unlink(missing_ok=True)

    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(db_path)
    from csurv.schema import create_tables
    create_tables(conn)

    # Insert products
    for i, (name, cat) in enumerate(PRODUCTS):
        conn.execute("INSERT INTO products VALUES (?,?,?)", (i + 1, name, cat))
    conn.commit()

    # Draw Dirichlet probabilities
    cf_probs = {k: rng.dirichlet(v["dirichlet_alpha"])
                for k, v in CUSTOMER_FEATURES.items()}
    vf_probs = {k: rng.dirichlet(v["dirichlet_alpha"])
                for k, v in VISIT_FEATURES.items()}

    # Simulation state
    queue = defaultdict(list)  # A queue for each week, tracking [(cid, visit number, previous a-adjustment, previous b-adjustment, visit date)]
    cust_feats = {}            # A dictionary matching customers to dicts containing {feat_name: value_or_None}
    cid_counter = 0
    vid_counter = 0

    # Build list of the start days (mondays) of each week
    mon = SIM_START - timedelta(days=SIM_START.weekday())
    weeks = []
    while mon <= SIM_END:
        weeks.append(mon)
        mon += timedelta(days=7)

    # Populate new customer arrivals into weekly queues
    for wk in weeks:
        season = get_season(wk)
        n_new = rng.poisson(SEASONAL_RATES[season]) #Determine the number of new customers

        for _ in range(n_new):
            cid_counter += 1 #Picks the next sequential customer ID
            cid = cid_counter

            dow = int(rng.choice(7, p=DOW_PROBS)) #Pick the day of week for the visit
            first_date = wk + timedelta(days=dow)
            if first_date > SIM_END:
                continue

            # Sample customer features
            feats = {} 
            for fname, fdef in CUSTOMER_FEATURES.items():
                val = sample_category(cf_probs[fname], fdef["values"], rng)
                feats[fname] = val  # may be None for optional features

            cust_feats[cid] = feats

            # Insert customer
            conn.execute("INSERT INTO customers VALUES (?,?)",
                         (cid, first_date.isoformat()))

            # Insert non-null features
            for fname, fval in feats.items():
                if fval is not None:
                    conn.execute(
                        "INSERT INTO customer_features VALUES (?,?,?)",
                        (cid, fname, fval))

            queue[wk].append((cid, 1, None, None, first_date))

    conn.commit()

    # Process visits week by week
    churn_dict = {}
    for wk in weeks:
        dish_features_adjs = {}
        for name, _cat in PRODUCTS:
            da, db = draw_adjustment(("dish", name), rng)
            dish_features_adjs[name] = (da, db)

        for (cid, visit_idx, a_prev, b_prev, forced_date) in queue.get(wk, []):

            # Determine visit date
            if forced_date is not None and visit_idx==1: #If the visit date was already selected, we don't need to pick it
                visit_date = forced_date
            else:
                dow = int(rng.choice(7, p=DOW_PROBS)) #For returning visitors, pick a date
                visit_date = wk + timedelta(days=dow)
                #If return visit is the same week and the day is randomized to too early
                if forced_date is not None and visit_date <= forced_date:
                    valid_days = [d for d in range(0,7) if wk+timedelta(days=d)>forced_date]
                    #If the previous visit was sunday, push to the next week
                    if not valid_days:
                        next_wk = wk + timedelta(weeks=1)
                        if next_wk <= weeks[-1]:
                            queue[next_wk].append((cid, visit_idx, a_prev, b_prev, forced_date))
                        continue
                    #If the previous visit was not sunday, randomize from the remaining days
                    else:
                        probs = DOW_PROBS[valid_days]
                        probs = probs / probs.sum()
                        dow = int(rng.choice(valid_days, p=probs))
                        visit_date = wk + timedelta(days=dow)

            if visit_date > SIM_END:
                continue

            vid_counter += 1 #track visit ID
            vid = vid_counter

            # Beta prior
            if visit_idx == 1:
                a_prior, b_prior = 1.0, 1.0
            else:
                churn_frac = b_prev / (a_prev + b_prev)
                a_prior = 1.0
                b_prior = 1.0 + (visit_idx - 1) * churn_frac



            #Visit features: Reservation Type
            reservation_type = sample_category(
                vf_probs["reservation_type"],
                VISIT_FEATURES["reservation_type"]["values"], rng)

            # Party size
            if reservation_type!="Private Event":
                party_size = int(rng.choice(np.arange(1, 8), p=PARTY_SIZE_PROBS))
            elif reservation_type=="Private Event":
                party_size = int(rng.choice(np.arange(6, 11), p=EVENT_SIZE_PROBS))



            # Visit features: Special offer (not all random)
            if is_special_day(visit_date,"Valentines"):
                special_offer = "Valentines Day"
            elif is_special_day(visit_date,"New Years Eve"):
                special_offer = "New Years"
            else:
                special_offer = sample_category(
                    vf_probs["special_offer_random"],
                    VISIT_FEATURES["special_offer_random"]["values"], rng)

            # Visit features: Discount
            discount_val = sample_category(
                vf_probs["discount"],
                VISIT_FEATURES["discount"]["values"], rng)
            if discount_val is None:
                partial_full, disc_reason = None, None
            else:
                parts = discount_val.split("+")
                partial_full = parts[0]
                disc_reason = parts[1]

            dow_name = visit_date.strftime("%A")

            # Insert visit
            conn.execute(
                "INSERT INTO visits VALUES (?,?,?,?,?,?,?,?,?)",
                (vid, cid, visit_date.isoformat(), party_size,
                 reservation_type, special_offer, dow_name,
                 partial_full, disc_reason))

            # Sample dishes
            entree_idx = ENTREE_IDX_BEFORE if visit_date < MENU_SWAP_DATE else ENTREE_IDX_AFTER
            n_app = int(rng.integers(0, party_size + 1))
            n_ent = party_size
            n_des = int(rng.integers(0, party_size + 1))

            ordered = []
            if n_app > 0:
                ordered.extend(rng.choice(APP_IDX, size=n_app, replace=True))
            ordered.extend(rng.choice(entree_idx, size=n_ent, replace=True))
            if n_des > 0:
                ordered.extend(rng.choice(DESSERT_IDX, size=n_des, replace=True))

            dish_counts = Counter(int(x) for x in ordered)
            dish_names = []
            for prod_idx, qty in dish_counts.items():
                conn.execute("INSERT INTO purchases (visit_id, product_id, quantity) VALUES (?,?,?)",
                             (vid, prod_idx + 1, qty))
                dish_names.append(PRODUCTS[prod_idx][0])

            # Beta adjustment: sum features + sum dishes, then mean of the two
            feat_a, feat_b = 0.0, 0.0
            feats = cust_feats[cid]
            for fname in CUSTOMER_FEATURES:
                val = feats.get(fname)
                da, db = draw_adjustment((fname, val), rng)
                feat_a += da
                feat_b += db

            # Visit feature adjustments
            for fname, val in [("reservation_type", reservation_type),
                               ("special_offer", special_offer),
                               ("discount", discount_val)]:
                da, db = draw_adjustment((fname, val), rng)
                feat_a += da
                feat_b += db

            dish_a, dish_b = 0.0, 0.0
            for dname in dish_names:
                da, db = dish_features_adjs[dname]
                dish_a += da
                dish_b += db
            #Divide by party_size to avoid over-weighting dish effects for large parties
            dish_a /= party_size
            dish_b /= party_size

            a_adj = (feat_a + dish_a) / 2.0
            b_adj = (feat_b + dish_b) / 2.0

            a_prime = a_prior + a_adj
            b_prime = b_prior + b_adj

            # Churn decision
            p_churn = rng.beta(a_prime, b_prime)
            churned = rng.random() < p_churn
            churn_dict[vid]=churned

            if not churned:
                lam = WEIBULL_BASE_SCALE * (a_prime + b_prime) / b_prime
                dw = max(0, int(np.round(rng.weibull(WEIBULL_SHAPE) * lam)))
                ret_wk = wk + timedelta(weeks=dw)
                if ret_wk <= weeks[-1]:
                    queue[ret_wk].append((cid, visit_idx + 1, a_prime, b_prime, visit_date))

        # Commit each week
        conn.commit()

    # Summary
    conn2 = sqlite3.connect(db_path)
    nc = conn2.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    nv = conn2.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
    np_ = conn2.execute("SELECT COUNT(*) FROM purchases").fetchone()[0]
    nf = conn2.execute("SELECT COUNT(*) FROM customer_features").fetchone()[0]
    conn2.close()
    with open("data/churn.json", "w", encoding="utf-8") as f:
            json.dump(churn_dict, f, indent=4)
    print(f"Generated synthetic data in {db_path}:")
    print(f"  {nc:,} customers")
    print(f"  {nv:,} visits")
    print(f"  {np_:,} purchases")
    print(f"  {nf:,} customer feature entries")
    print(f"  Simulation: {SIM_START} to {SIM_END}")


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "demo.db"
    generate_data(db)
