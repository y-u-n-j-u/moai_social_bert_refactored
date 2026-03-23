import os
import sys
import numpy as np
import pandas as pd

from .dataset import moai_social_bertDataset
from .util import is_target_outbound
sys.path.append(os.path.realpath('./dataset'))

class JRDBDataset(moai_social_bertDataset):
    def __init__(self, split, args):
        super().__init__(split, args)
        filepath = os.path.join(self.path, split + '_trajnet.pkl')
        df_data = pd.read_pickle(filepath)
        df_data.head()
        self.env = {}
        self.scales = {
        'bytes-cafe-2019-02-07_0': 1.0,
        'clark-center-2019-02-28_0': 1.0,
        'clark-center-2019-02-28_1': 1.0,
        'clark-center-intersection-2019-02-28_0': 1.0,
        'cubberly-auditorium-2019-04-22_0': 1.0,
        'forbes-cafe-2019-01-22_0': 1.0,
        'gates-159-group-meeting-2019-04-03_0': 1.0,
        'gates-ai-lab-2019-02-08_0': 1.0,
        'gates-basement-elevators-2019-01-17_1': 1.0,
        'gates-to-clark-2019-02-28_1': 1.0,
        'hewlett-packard-intersection-2019-01-24_0': 1.0,
        'huang-2-2019-01-25_0': 1.0,
        'huang-basement-2019-01-25_0': 1.0,
        'huang-lane-2019-02-12_0': 1.0,
        'jordan-hall-2019-04-22_0': 1.0,
        'memorial-court-2019-03-16_0': 1.0,
        'meyer-green-2019-03-16_0': 1.0,
        'nvidia-aud-2019-04-18_0': 1.0,
        'packard-poster-session-2019-03-20_0': 1.0,
        'packard-poster-session-2019-03-20_1': 1.0,
        'packard-poster-session-2019-03-20_2': 1.0,
        'stlc-111-2019-04-19_0': 1.0,
        'svl-meeting-gates-2-2019-04-08_0': 1.0,
        'svl-meeting-gates-2-2019-04-08_1': 1.0,
        'tressider-2019-03-16_0': 1.0,
        'tressider-2019-03-16_1': 1.0,
        'tressider-2019-04-26_2': 1.0
        }

        self.min_obs_len = self.args.min_obs_len if self.split == 'train' else self.args.obs_len

        scene_trajs, meta, scene_ids, scene_frames, scene_start_frames = self.split_trajectories_by_scene(df_data, self.args.obs_len+self.args.pred_len)
        for trajs, scene_id, frames, start_frames in zip(scene_trajs, scene_ids, scene_frames, scene_start_frames):
            for start_frame in start_frames:
                curr_trajs = trajs[frames == start_frame]
                for idx in range(len(curr_trajs)):
                    tmp_curr_trajs = curr_trajs.copy()
                    if idx != 0:
                        tmp_curr_trajs[[0, idx]] = tmp_curr_trajs[[idx, 0]]
                    if is_target_outbound(tmp_curr_trajs[0], self.args.obs_len, traj_bound=self.args.view_range):
                        print("target is outbound")
                        continue
                    self.all_trajs.append(tmp_curr_trajs)
                    self.all_scenes.append(scene_id)

            if self.split == 'test' and self.args.viz and len(self.all_trajs) > 100:
                break


    def split_trajectories_by_scene(self, data, total_len):
        trajectories = []
        meta = []
        scene_list = []
        scene_frames = []
        first_frames = []
        for meta_id, meta_df in data.groupby('sceneId', as_index=False):
            frames = meta_df[['frame']].to_numpy().astype('float32').reshape(-1, total_len)
            frames = np.min(frames, axis=-1)
            first_frames.append(frames)
            scene_frames.append(np.unique(frames))
            trajectories.append(meta_df[['x', 'y']].to_numpy().astype('float32').reshape(-1, total_len, 2))
            meta.append(meta_df)
            scene_list.append(meta_id)

        return np.array(trajectories), meta, scene_list, first_frames, scene_frames
    
