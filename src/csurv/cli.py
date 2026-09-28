import click

@click.group()

def main():
    """Customer Churn / Lifetime Value Pipeline"""
    pass

@main.command()
@click.option("--db","db_path",
              default="demo.db",
              help="Path to SQLite database to be created")
@click.option("--seed","seed",
              default=42,
              type=int,
              help="Random Seed")

def generate(db_path,seed):
    """Generates synthetic data"""
    from csurv.generate import generate_data
    generate_data(db_path,seed)
    click.echo(f"Generated synthetic data in {db_path}")

@main.command()
@click.option("--db","db_path",required=True,help="Path to SQLite database.")
@click.option("--out","out_path",default="models/",help="Directory to save trained model.")
def train(db_path,out_path):
    """Train model and save output"""
    from csurv.model import EMCure_Learn
    import json
    import os

    # Model
    betas, mean_sd = EMCure_Learn(db_path)
    output = {"betas": {k: v.tolist() for k, v in betas.items()},
              "mean_sd": mean_sd
              }
    #    Save to file
    os.makedirs(out_path, exist_ok=True)
    with open(out_path+"/demo.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    click.echo(f"Training on {db_path}, saving to {out_path}")

@main.command()
@click.option("--db", "db_path",required=True, help="Path to SQLite database.")
@click.option("--model", "model_path",default="models/demo.json", help="Path to trained model JSON.")
@click.option("--out", "out_path",default="predictions.csv", help="Output predictions file.")
@click.option("--horizon", default=90, type=int, help="Prediction horizon in days.")
@click.option("--ticket", default=65.0, type=float, help="Average ticket price ($).")
@click.option("--churn-reduction-factor", "churn_reduction_factor",
              default=1.0, type=float,
              help="Multiplier on pi (0.90 = 10%% churn reduction).")
def predict(db_path, model_path, out_path, horizon, ticket, churn_reduction_factor):
    """Score customers using a trained model."""
    from csurv.predict import predict as run_predict
    results = run_predict(db_path, model_path, horizon, ticket, churn_reduction_factor)
    results.to_csv(out_path)
    click.echo(f"Saved {len(results)} predictions to {out_path}")

@main.command()
def demo():
    """Run generate -> train -> predict end-to-end on synthetic data."""
    click.echo("Running full demo pipeline...")
    ctx = click.get_current_context()
    ctx.invoke(generate, db_path="data/demo.db", seed=42)
    ctx.invoke(train, db_path="data/demo.db", out_path="models/")
    ctx.invoke(predict, db_path="data/demo.db", model_path="models/demo.json",
               out_path="predictions.csv", horizon=90, ticket=65.0,
               churn_reduction_factor=0.90)


    