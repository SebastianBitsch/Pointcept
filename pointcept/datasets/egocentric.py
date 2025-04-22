import sys
import os
import numpy as np
import torch
from pathlib import Path

from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
from .preprocessing.scannet.meta_data.scannet200_constants import VALID_CLASS_IDS_200

sys.path.append(os.path.abspath("02501-Ego3D/src"))

from ego3d.data.egocentric_view_generator import create_360_view
from ego3d.data.camera_poses import extract_poses, subsample_poses
from ego3d.data.pcl_ground_truth_generator import process_mesh
from ego3d.paths import iterate_scannet

@DATASETS.register_module()
class EgoCentricScanNet200Dataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment200",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_200)

    def __init__(
        self,
        lr_file=None,
        la_file=None,
        n_views_per_scene=10,
        view_size=(1024, 512),
        max_dist=100.0,
        **kwargs,
    ):
        self.n_views_per_scene = n_views_per_scene
        self.view_size = view_size
        self.max_dist = max_dist
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = torch.load(la_file) if la_file is not None else None
        
        super().__init__(**kwargs)
    
    def get_data_list(self):
        if self.lr is None:
            data_list = []
            for raw_dir, processed_dir in iterate_scannet():
                scene_id = raw_dir.name
                for view_idx in range(self.n_views_per_scene):
                    data_list.append((scene_id, raw_dir, processed_dir, view_idx))
            
            return data_list
        else:
            return [
                (scene_id, raw_dir, processed_dir, view_idx)
                for scene_id in self.lr
                for raw_dir, processed_dir in iterate_scannet(base_dir=self.data_root)
                if raw_dir.name == scene_id
                for view_idx in range(self.n_views_per_scene)
            ]

    def get_split_name(self, idx):
        """ Kinda hacky tbh"""
        scene_id, raw_dir, processed_dir, view_idx = self.data_list[idx % len(self.data_list)]
        return "test" if "/scans_test/" in str(processed_dir) else "train"

    def get_data_name(self, idx):
        out = self.data_list[idx % len(self.data_list)]
        scene_id, _, _, view_idx = out
        full_out = f"{scene_id}_view{view_idx}"
        return full_out
    
    def get_data(self, idx):
        a = idx % len(self.data_list)
        scene_id, raw_dir, processed_dir, view_idx = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        
        if self.cache:
            cache_name = f"pointcept-{name}"
            cached_data = shared_dict(cache_name)
            if cached_data is not None:
                return cached_data
        
        os.makedirs(processed_dir, exist_ok=True)
        
        # Generate point cloud
        gt_mesh_path = processed_dir / f"{scene_id}_gt.ply"
        if not gt_mesh_path.exists():
            process_mesh(
                raw_dir / f"{scene_id}_vh_clean.ply", 
                gt_mesh_path,
                label_path=raw_dir / f"{scene_id}_vh_clean.labels.ply",
                segs_path=raw_dir / f"{scene_id}_vh_clean.segs.json",
                agg_path=raw_dir / f"{scene_id}_vh_clean.aggregation.json",
                debug=False
            )
                
        poses_file = extract_poses(raw_dir, processed_dir)
        transform_matrix = subsample_poses(poses_file, samples=self.n_views_per_scene)[view_idx]
        camera_pos = transform_matrix[:3, 3]
        
        _, point_cloud = create_360_view(
            mesh_path=gt_mesh_path, 
            viewpoint=camera_pos,
            width=self.view_size[0], 
            height=self.view_size[1],
            max_dist=self.max_dist, 
            debug=False
        )
        data_dict = {
            'color': np.asarray(point_cloud.colors, dtype=np.float32),
            'coord': np.asarray(point_cloud.points, dtype=np.float32),
            'normal': np.asarray(point_cloud.normals, dtype=np.float32),
            'segment200': np.asarray(point_cloud.semantic_id, dtype=np.int32),
            'instance': np.asarray(point_cloud.instance_id, dtype=np.int32),
            'name': name,
            'split': split
        }
        print("data dict", [(k, v.shape) for k,v in data_dict.items()])
        
        # Convert from segment200 to segment
        data_dict["segment"] = data_dict.pop("segment200").reshape([-1]).astype(np.int32)
        
        if self.la is not None and name in self.la:
            sampled_index = self.la[name]
            mask = np.ones_like(data_dict["segment"], dtype=bool)
            mask[sampled_index] = False
            data_dict["segment"][mask] = self.ignore_index
            data_dict["sampled_index"] = sampled_index
        
        if self.cache:
            shared_dict(cache_name, data_dict)
        
        return data_dict