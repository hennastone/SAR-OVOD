"""
Baseline A ve B'nin ortak tahmin dongusu.

Neden ayri modul: ultralytics'e 1500+ goruntuluk tek bir kaynak listesi verildiginde
Results nesneleri (her biri 4K orig_img tasiyor, ~24MB) birikip RAM'i sisiriyor ve
GPU bos kaliyor. Burada goruntuler kucuk parcalar halinde islenip her parca sonrasi
serbest birakiliyor - bellek parca boyutuyla sinirli kaliyor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm


def run_chunked(
    model,
    paths: list[Path],
    *,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    chunk: int = 16,
    desc: str = "predict",
) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict]]:
    """Her goruntu icin (path, xyxy, conf, cls, speed) uretir.

    Kutu yoksa bos diziler doner - cagiran taraf uzunluk kontrolu yapmali.

    Not: res.path toplu cagride orijinal dosya adini korumuyor ('image0.jpg' gibi
    donuyor), bu yuzden yol bilgisi kendi girdi listemizden eslestirilir. Sonuclar
    girdi sirasini korur; uzunluk esitligi ayrica kontrol edilir.
    """
    for start in tqdm(range(0, len(paths), chunk), desc=desc, unit="chunk"):
        batch_paths = paths[start : start + chunk]
        batch = [str(p) for p in batch_paths]
        results = model.predict(
            source=batch,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            stream=False,
            verbose=False,
        )
        if len(results) != len(batch_paths):
            raise RuntimeError(
                f"Sonuc sayisi girdiyle uyusmuyor ({len(results)} != {len(batch_paths)}); "
                "yol eslestirmesi guvenilir degil"
            )
        for src_path, res in zip(batch_paths, results):
            b = res.boxes
            if b is None or len(b) == 0:
                empty = np.empty((0, 4), dtype=np.float32)
                yield str(src_path), empty, np.empty(0, np.float32), np.empty(0, np.int64), res.speed
            else:
                yield (
                    str(src_path),
                    b.xyxy.cpu().numpy(),
                    b.conf.cpu().numpy(),
                    b.cls.cpu().numpy().astype(int),
                    res.speed,
                )
        # Results referanslarini birakip bir sonraki parcaya gec
        del results
