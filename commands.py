import fire

from covid_tweet_analysis.infer import infer
from covid_tweet_analysis.test import test_model
from covid_tweet_analysis.train import train


def main():
    fire.Fire(
        {
            "train": train,
            "test": test_model,
            "infer": infer,
        }
    )


if __name__ == "__main__":
    main()
