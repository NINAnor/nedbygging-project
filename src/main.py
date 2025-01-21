import logging
import pathlib

import environ
import hydra

env = environ.Env()
BASE_DIR = pathlib.Path(__file__).parent.parent
environ.Env.read_env(str(BASE_DIR / ".env"))

DEBUG = env.bool("DEBUG", default=False)

logging.basicConfig(level=(logging.DEBUG if DEBUG else logging.INFO))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg) -> None:
    logging.info("Hello World!")
    logging.debug(f"Configuration: {cfg}")


if __name__ == "__main__":
    main()
