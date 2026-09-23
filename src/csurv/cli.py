import click

@click.group()

def main():
    """Customer Churn / Lifetime Value Pipeline"""
    pass

@main.command()
@click.option("--db","db_path",
              default="demo.db",
              help="Path to SQLite database to be created")

def generate(db_path):
    """Generates synthetic data"""
    from csurv.generate import generate_data
    generate_data(db_path)
    click.echo(f"Generated synthetic data in {db_path}")

@main.command()
@click.option("--db","db_path",required=True,help="Path to SQLite database.")
@click.option("--out","out_path",default="models/",help="Directory to save trained model.")
def train(db_path,out_path):
    """Train model and save output"""
    click.echo(f"Training on {db_path}, saving to {out_path}")

@main.command()
@click.option("--db", "db_path",required=True, help="Path to SQLite database.")
@click.option("--model", "model_path",default="models/", help="Path to trained model directory.")
@click.option("--out", "out_path",default="predictions.csv", help="Output predictions file.")
def predict(db_path, model_path, out_path):
    """Score customers using a trained model."""
    click.echo(f"Predicting from {db_path} using {model_path}, saving to {out_path}")

@main.command()
def demo():
    """Run generate -> train -> predict end-to-end on synthetic data."""
    click.echo("Running full demo pipeline...")


    