from pathlib import Path

import kagglehub


HANDLE = "itshpark/data-driven-prediction-of-battery-cycle/versions/1"
FILES = [
    "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
]


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        print(f"Downloading {filename}", flush=True)
        path = kagglehub.dataset_download(
            HANDLE,
            path=filename,
            output_dir=str(output_dir),
        )
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
