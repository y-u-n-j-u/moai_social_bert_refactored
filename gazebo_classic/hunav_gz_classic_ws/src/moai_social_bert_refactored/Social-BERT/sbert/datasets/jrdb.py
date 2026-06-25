import os
import numpy as np
import pandas as pd

from .dataset import moai_social_bertDataset
from .util import is_target_outbound


def subsample_raw90_to_model(xy_raw: np.ndarray, obs_len: int, pred_len: int, stride: int = 3) -> np.ndarray:
    """
    xy_raw: [90,2] where raw_obs=obs_len*stride, raw_pred=pred_len*stride
    Example obs_len=10,pred_len=20,stride=3 -> raw_obs=30, raw_pred=60, total=90

    Returns: [obs_len+pred_len,2] -> [30,2]
      obs  : 0,3,6,...,27
      pred : 30,33,...,87
    """
    raw_obs = obs_len * stride
    raw_pred = pred_len * stride
    raw_total = raw_obs + raw_pred
    if xy_raw.shape[0] < raw_total:
        raise ValueError(f"xy_raw length={xy_raw.shape[0]} < raw_total={raw_total}")

    obs = xy_raw[:raw_obs:stride]
    pred = xy_raw[raw_obs:raw_obs + raw_pred:stride]
    return np.concatenate([obs, pred], axis=0)


class JRDBDataset(moai_social_bertDataset):
    """
    Expects preprocessed pkl with columns:
      sceneId, metaId, trackId, frame, x, y

    Each (sceneId, metaId, trackId) should contain 90 contiguous frames already.
    Loader only:
      - groups by (sceneId, metaId)
      - for each track takes first 90 rows (sorted by frame)
      - converts raw 90 -> model 30 using stride=3 (default)
      - creates one sample per target by swapping to index 0
    """

    def __init__(self, split, args):
        super().__init__(split, args)

        # where trainer expects data:
        # self.path = os.path.join(args.dataset_path, args.dataset_name)  (base class)
        filepath = os.path.join(self.path, f"{split}_trajnet.pkl")
        df = pd.read_pickle(filepath)
        
        # ---- REQUIRED for base dataset.__getitem__ ----
        # dataset.py expects self.scales[scene_id] to exist.
        default_scale = float(getattr(args, "traj_scale", getattr(args, "global_scale", 1.0)))
        self.scales = {sid: default_scale for sid in df["sceneId"].unique()}


        required = {"sceneId", "metaId", "trackId", "frame", "x", "y"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"[JRDBDataset] Missing columns in pkl: {missing}")

        # stride for downsampling inside each meta-window
        # This is not exposed in the trainer argparse path, so use getattr with a default of 3.
        stride = int(getattr(args, "subsample_stride", 3))

        # raw length expected from obs/pred and stride
        raw_total = int(args.obs_len) * stride + int(args.pred_len) * stride  # e.g., 90

        df = (df.sort_values(["sceneId", "metaId", "trackId", "frame"])
                .drop_duplicates(subset=["sceneId", "metaId", "trackId", "frame"], keep="first")
                .reset_index(drop=True))

        self.all_trajs = []
        self.all_scenes = []

        # build per (sceneId, metaId)
        for (scene_id, meta_id), df_m in df.groupby(["sceneId", "metaId"], sort=False):
            trajs = []

            # each track in this meta-window
            for tid, tdf in df_m.groupby("trackId", sort=False):
                tdf = tdf.sort_values("frame")
                if len(tdf) < raw_total:
                    continue

                tdf = tdf.iloc[:raw_total]
                xy_raw = tdf[["x", "y"]].to_numpy(dtype=np.float32)  # [90,2] expected

                xy_model = subsample_raw90_to_model(
                    xy_raw=xy_raw,
                    obs_len=int(args.obs_len),
                    pred_len=int(args.pred_len),
                    stride=stride
                )  # [obs+pred,2] -> [30,2]

                trajs.append(xy_model)

            if len(trajs) == 0:
                continue

            trajs = np.stack(trajs, axis=0)  # [N, seq_len, 2]

            # Social-BERT: one sample per target by swapping to index 0
            for idx in range(len(trajs)):
                tmp = trajs.copy()
                if idx != 0:
                    tmp[[0, idx]] = tmp[[idx, 0]]

                # keep original outbound logic
                if is_target_outbound(tmp[0], int(args.obs_len), traj_bound=float(args.view_range)):
                    continue

                self.all_trajs.append(tmp)
                self.all_scenes.append(scene_id)

            if split == "test" and getattr(args, "viz", False) and len(self.all_trajs) > 100:
                break

        # expose
        self.trajs = self.all_trajs
        self.scenes = self.all_scenes

        # DEBUG: if the sample count drops to zero, this makes the failure point obvious.
        print(f"[JRDBDataset] loaded samples: {len(self.trajs)} (split={split})")

    def __len__(self):
        return len(self.trajs)
