import argparse
import csv
from pathlib import Path


def read_map_from_csv(csv_path: Path, key_col: str, val_col: str) -> dict:
    m = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            k = row.get(key_col)
            v = row.get(val_col)
            if k is None or v is None:
                continue
            m[k] = v
    return m


def main():
    ap = argparse.ArgumentParser()
    base = Path.cwd()
    ap.add_argument(
        "--det-csv",
        type=str,
        default=str(base / "runs_yolo" / "infer_test_dataset_gridBretrain5" / "results.csv"),
    )
    ap.add_argument(
        "--det-col",
        type=str,
        default="tumor",
        help="Column name for value in detection CSV",
    )
    ap.add_argument(
        "--cam-csv",
        type=str,
        default=str(base / "runs_cls" / "cla_pre_test_dataset.csv"),
    )
    ap.add_argument(
        "--cam-col",
        type=str,
        default="pred",
        help="Column name for value in CAM/Seg CSV",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(base / "runs_combined" / "combined.csv"),
    )
    args = ap.parse_args()

    det_csv = Path(args.det_csv)
    cam_csv = Path(args.cam_csv)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("Reading detection:", det_csv)
    det = read_map_from_csv(det_csv, key_col="filename", val_col=args.det_col)
    print("Detection entries:", len(det))

    print("Reading CAM/Seg:", cam_csv)
    cam = read_map_from_csv(cam_csv, key_col="filename", val_col=args.cam_col)
    print("CAM/Seg entries:", len(cam))

    fnames = sorted(set(det.keys()) | set(cam.keys()))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "yolo_tumor", "seg_tumor", "combined_tumor"])
        for fn in fnames:
            yolo = det.get(fn, "no")
            seg_pred = cam.get(fn, "no") # Default to 'no' instead of '0'
            
            # Logic for combination: if either says yes, then yes
            # Note: seg_pred might be 'yes'/'no' or '1'/'0' depending on source
            is_seg_pos = (seg_pred.lower() == "yes" or seg_pred == "1")
            is_yolo_pos = (yolo.lower() == "yes" or yolo == "1")
            
            combined = "yes" if (is_yolo_pos or is_seg_pos) else "no"
            w.writerow([fn, yolo, seg_pred, combined])

    print("Combined CSV:", out_csv)
    from collections import Counter

    cnt = Counter()
    with open(out_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cnt[row["combined_tumor"]] += 1
    print("Counts:", dict(cnt))


if __name__ == "__main__":
    main()

