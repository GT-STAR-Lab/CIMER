import click
import joblib

@click.command(help='Help')
@click.option('--filename', type=str, help='Path to trial data archive', required=True)
def main(filename):
    trials = joblib.load(filename)
    print(trials.losses())
    print(trials.best_trial)

if __name__ == '__main__':
    main()