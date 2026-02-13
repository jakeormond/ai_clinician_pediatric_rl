"""
Creates DataProcessor class to process data stored in HDF5 format.
All normalization and transformation is done here.
"""
import yaml
import copy
import h5py
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import time
import pyarrow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

import torch
from torch.utils.data import Dataset, DataLoader

import logging

from src import config_manager
logger = logging.getLogger(__name__)

class ShardProcessor:
    def __init__(
            self, 
            config_manager, 
            split = 'train', 
            normalize = True,
            norm_params=None, 
        ):
        """
        Unlike with dataframes, with the shards, I've already dropped the features we
        don't want to use, so we just need to normalize, create binary actions,
        and then split into the appropriate dataset format.
        Note that features have already been transformed as well
        -------------
        
        """
        self.split = split

        if self.split == 'train' and config_manager.training_config['tuning']:
            self.split = 'tuning'

        # find the shards for the given split
        if config_manager.split_num is None:
            shard_dir = os.path.join(config_manager.paths['data_dir'], 
                        config_manager.data_config['shard_dir'])
        else:
            shard_dir = os.path.join(config_manager.paths['data_dir'], 
                        config_manager.data_config['shard_dir'], 
                        f"split_{config_manager.split_num}")

        if not config_manager.training_config['tuning']:
            self.shard_dir = os.path.join(shard_dir, 'training_set')
        else:
            self.shard_dir = os.path.join(shard_dir, 'tuning_set')
        
        final_file_name = f"{self.split}_dynamic_deephit_dataset.h5"
        final_file_path = os.path.join(self.shard_dir, final_file_name)

        # load if exists
        if os.path.exists(final_file_path):
            logger.info(f"Loading existing dynamic deephit dataset from {final_file_path}")
            with h5py.File(final_file_path, 'r') as f:
                self.dynamic_deephit_dataset = {}
                self.dynamic_deephit_dataset['feature_names'] = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]
                self.dynamic_deephit_dataset['stay_ids'] = f['stay_ids'][:]
                self.dynamic_deephit_dataset['index_ptr'] = f['index_ptr'][:]
                self.dynamic_deephit_dataset['trajectories'] = f['trajectories'][:].astype(np.float32)
                self.dynamic_deephit_dataset['event'] = f['event'][:].astype(np.int32)
                self.dynamic_deephit_dataset['time_to_event_raw'] = f['time_to_event_raw'][:].astype(np.int32)
                self.dynamic_deephit_dataset['time_to_event'] = f['time_to_event'][:].astype(np.int32)

                self.physiological_feature_ind = f['physiological_feature_ind'][:].astype(np.int32).tolist()
                self.physiological_feature_type = [f['physiological_feature_type'][i].decode('utf-8') for i in range(len(f['physiological_feature_type']))]

                self.feature_dim = f['feature_dim'][()]

                if 'predictions' in f:
                    self.dynamic_deephit_dataset['predictions'] = f['predictions'][:].astype(np.float32)

                if config_manager._data_config['cohort'] is not None and config_manager._data_config['cohort'] != 'all':
                    # filter the dataset to only include the specified cohort
                    logger.info(f"Filtering dynamic deephit dataset to only include cohort: {config_manager._data_config['cohort']}")
                    self.filter_cohort(config_manager)

            return

        self.norm_params = norm_params
        self.window_length = config_manager.training_config['window_length']
        self.time_bins = config_manager.time_bins

        self.max_time = self.time_bins[-1]

        # load features to use from config_manager
        self.features_to_use = pd.read_csv(config_manager.data_config['features_to_use_path'])

        
        # if config_manager.action_mode == 'binary_only' or config_manager.action_mode == 'mixed':
        #     create_binary_actions = True
        # else:
        #     create_binary_actions = False            

        # if config_manager.action_mode == 'binary_only':
        #     self.normalize_actions = False
        #     self.transform_actions = False                 

        all_files = os.listdir(self.shard_dir)                
        
        # raw_shard_files = [os.path.join(self.shard_dir, f) for f in all_files if 
        #                         f.startswith(self.split) and f.endswith("_shard.h5")]
        raw_shard_files = [os.path.join(self.shard_dir, f) for f in all_files if 
                                f.startswith(self.split) and f.endswith(".h5")]
        if len(raw_shard_files) == 0:
            raise ValueError(f"No shard files found for split {self.split} in directory {self.shard_dir}")

        # sort the files to ensure consistent order
        self.raw_shard_files = sorted(raw_shard_files)

        
        ########### Normalized Shards - we remove long trajectories first ##############
        normalized_shard_files = [os.path.join(self.shard_dir, f) for f in all_files if 
                        f.startswith(self.split) and f.endswith("_normalized.h5")]
        # sort the files to ensure consistent order
        self.normalized_shard_files = sorted(normalized_shard_files)

        if len(self.normalized_shard_files) == 0: # i.e. the files have not been normalized yet
            self.normalize(config_manager=config_manager)
            all_files = os.listdir(self.shard_dir)

            normalized_shard_files = [os.path.join(self.shard_dir, f) for f in all_files if 
                                f.startswith(self.split) and f.endswith("_normalized.h5")]
            
            # sort the files to ensure consistent order
            self.normalized_shard_files = sorted(normalized_shard_files)        
        
        # create dynamic deephit dataset
        self.dynamic_deephit_dataset = {}
        for i, shard_file in enumerate(self.normalized_shard_files):
            logger.info(f"Loading normalized shard file into memory for tuning: {shard_file}")
            # load the shard file
            with h5py.File(shard_file, 'r') as f:      
                # convert feature_names from bytes to strings
                feature_names = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]                
                stay_ids = f['stay_ids'][:]
                index_ptr = f['index_ptr'][:]
                trajectories = f['trajectories'][:]      
                event = f['event'][:]    

            final_indices = index_ptr[1:] - 1    
            final_events = event[final_indices].astype(int)

            # create the time_to_event array
            time_to_event = np.zeros_like(trajectories[:,0], dtype=np.int32)
            for j in range(len(stay_ids)):
                start_idx = index_ptr[j]
                end_idx = index_ptr[j+1]
                length = end_idx - start_idx
                tte = np.arange(length, 0, -1)
                time_to_event[start_idx:end_idx] = tte


            if i == 0:
                self.dynamic_deephit_dataset['feature_names'] = [name.encode('utf-8') for name in feature_names]
                self.dynamic_deephit_dataset['stay_ids'] = stay_ids
                self.dynamic_deephit_dataset['index_ptr'] = index_ptr
                self.dynamic_deephit_dataset['trajectories'] = trajectories.astype(np.float32)
                self.dynamic_deephit_dataset['event'] = final_events.astype(np.int32)
                self.dynamic_deephit_dataset['time_to_event_raw'] = time_to_event.astype(np.int32)
            else:
                # raise not implemented error, I'm not handling index_ptr concatenation here
                raise NotImplementedError("Merging multiple shard files into dynamic_deephit_dataset not implemented yet.")
                # self.dynamic_deephit_dataset['stay_ids'] = np.concatenate(
                #     (self.dynamic_deephit_dataset['stay_ids'], stay_ids), axis=0)
                # self.dynamic_deephit_dataset['index_ptr'] = np.concatenate(
                #     (self.dynamic_deephit_dataset['index_ptr'], 
                #         index_ptr[1:] + self.dynamic_deephit_dataset['index_ptr'][-1]), axis=0)
                # self.dynamic_deephit_dataset['trajectories'] = np.concatenate(
                #     (self.dynamic_deephit_dataset['trajectories'], trajectories.astype(np.float32)), axis=0)

        # bin time_to_events
        tte_bin = np.digitize(self.dynamic_deephit_dataset['time_to_event_raw'], self.time_bins, right=True) - 1                
        self.dynamic_deephit_dataset['time_to_event'] = tte_bin

        # get physiological feature indices for dynamic deephit model, and feature_dim
        # load 'feature_names' from the first h5 file
        with h5py.File(self.normalized_shard_files[0], 'r') as f:
            feature_names = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))] 
            self.feature_dim = f['trajectories'].shape[1]

        self.physiological_feature_ind = [] # for dynamic deephit model, which tries to predict next physiological features
        self.physiological_feature_type = []

        physiological_feature_sets = [
            'hemodynamics', 'labs', 'neurological', 'routine_vital_signs'Home (9A Sunnyside Rd S)
            ]
        
        for i, feature in enumerate(feature_names):
            if feature == 'time':
                continue
            
            if feature in self.features_to_use['feature'].values:
                feature_row = self.features_to_use[self.features_to_use['feature'] == feature]            
                feature_set = feature_row['feature_set'].values[0]
                data_type = feature_row['data_type'].values[0]
                if feature_set in physiological_feature_sets:
                    self.physiological_feature_ind.append(i)
                    self.physiological_feature_type.append(data_type)

        final_file_name = f"{self.split}_dynamic_deephit_dataset.h5"
        final_file_path = os.path.join(self.shard_dir, final_file_name)

        # save the dynamic deephit dataset to h5 file for future use
        self.feature_dim = self.dynamic_deephit_dataset['trajectories'].shape[1]
        self.dynamic_deephit_dataset_to_h5(final_file_path)
        
        # with h5py.File(final_file_path, 'w') as f:
        #     f.create_dataset('feature_names', data=self.dynamic_deephit_dataset['feature_names'])
        #     f.create_dataset('stay_ids', data=self.dynamic_deephit_dataset['stay_ids'], dtype=np.int32)
        #     f.create_dataset('index_ptr', data=self.dynamic_deephit_dataset['index_ptr'], dtype=np.int32)
        #     f.create_dataset('trajectories', data=self.dynamic_deephit_dataset['trajectories'], dtype=np.float32, 
        #                      chunks=(1, self.dynamic_deephit_dataset['trajectories'].shape[1]))
        #     f.create_dataset('event', data=self.dynamic_deephit_dataset['event'], dtype=np.int32)
        #     f.create_dataset('time_to_event_raw', data=self.dynamic_deephit_dataset['time_to_event_raw'], dtype=np.int32)
        #     f.create_dataset('time_to_event', data=self.dynamic_deephit_dataset['time_to_event'], dtype=np.int32)

        #     f.create_dataset('physiological_feature_ind', data=np.array(self.physiological_feature_ind, dtype=np.int32), dtype=np.int32)
        #     f.create_dataset('physiological_feature_type', data=np.array(self.physiological_feature_type, dtype='S20'), dtype='S20')

        #     f.create_dataset('feature_dim', data=np.array(self.feature_dim, dtype=np.int32), dtype=np.int32)

        # logger.info(f"Saved dynamic deephit dataset to {final_file_path}")

        if config_manager._data_config['cohort'] is not None and config_manager._data_config['cohort'] != 'all':
            # filter the dataset to only include the specified cohort
            logger.info(f"Filtering dynamic deephit dataset to only include cohort: {config_manager._data_config['cohort']}")
            self.filter_cohort(config_manager)

    
    def dynamic_deephit_dataset_to_h5(self, file_path):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('feature_names', data=self.dynamic_deephit_dataset['feature_names'])
            f.create_dataset('stay_ids', data=self.dynamic_deephit_dataset['stay_ids'], dtype=np.int32)
            f.create_dataset('index_ptr', data=self.dynamic_deephit_dataset['index_ptr'], dtype=np.int32)
            f.create_dataset('trajectories', data=self.dynamic_deephit_dataset['trajectories'], dtype=np.float32, 
                             chunks=(1, self.dynamic_deephit_dataset['trajectories'].shape[1]))
            f.create_dataset('event', data=self.dynamic_deephit_dataset['event'], dtype=np.int32)
            f.create_dataset('time_to_event_raw', data=self.dynamic_deephit_dataset['time_to_event_raw'], dtype=np.int32)
            f.create_dataset('time_to_event', data=self.dynamic_deephit_dataset['time_to_event'], dtype=np.int32)

            if hasattr(self, 'physiological_feature_ind'):
                f.create_dataset('physiological_feature_ind', data=np.array(self.physiological_feature_ind, dtype=np.int32), dtype=np.int32)
            if hasattr(self, 'physiological_feature_type'):
                f.create_dataset('physiological_feature_type', data=np.array(self.physiological_feature_type, dtype='S20'), dtype='S20')
            if hasattr(self, 'feature_dim'):
                f.create_dataset('feature_dim', data=np.array(self.feature_dim, dtype=np.int32), dtype=np.int32)
            
            if 'predictions' in self.dynamic_deephit_dataset:
                f.create_dataset('predictions', data=self.dynamic_deephit_dataset['predictions'], dtype=np.float32)
            if 'entropy' in self.dynamic_deephit_dataset:
                f.create_dataset('entropy', data=self.dynamic_deephit_dataset['entropy'], dtype=np.float32)
            if 'health_score' in self.dynamic_deephit_dataset:
                f.create_dataset('health_score', data=self.dynamic_deephit_dataset['health_score'], dtype=np.float32)


        logger.info(f"Saved dynamic deephit dataset to {file_path}")


    def filter_by_stay_ids(self, stay_ids_to_keep, config_manager):
        # filter the dynamic_deephit_dataset
        stay_ids = self.dynamic_deephit_dataset['stay_ids']
        logger.info(f"Original number of stays in dynamic_deephit_dataset: {len(stay_ids)}")
        logger.info(f"Original number of timesteps in dynamic_deephit_dataset: {self.dynamic_deephit_dataset['trajectories'].shape[0]}")

        # Create boolean mask for stays to keep
        mask = np.isin(stay_ids, list(stay_ids_to_keep))
        indices_to_keep = np.where(mask)[0]
        
        # logger.info(f"Number of stays after filtering to cohort {cohort}: {len(indices_to_keep)}")
        logger.info(f"Number of stays after filtering by stay ids: {len(indices_to_keep)}")


        # Build trajectory mask
        traj_mask = np.zeros(self.dynamic_deephit_dataset['trajectories'].shape[0], dtype=bool)
        for idx in indices_to_keep:
            start_idx = self.dynamic_deephit_dataset['index_ptr'][idx]
            end_idx = self.dynamic_deephit_dataset['index_ptr'][idx + 1]
            traj_mask[start_idx:end_idx] = True

        # Apply masks to all arrays at once
        self.dynamic_deephit_dataset['stay_ids'] = self.dynamic_deephit_dataset['stay_ids'][mask]
        self.dynamic_deephit_dataset['event'] = self.dynamic_deephit_dataset['event'][mask]
        self.dynamic_deephit_dataset['trajectories'] = self.dynamic_deephit_dataset['trajectories'][traj_mask]
        self.dynamic_deephit_dataset['time_to_event_raw'] = self.dynamic_deephit_dataset['time_to_event_raw'][traj_mask]
        self.dynamic_deephit_dataset['time_to_event'] = self.dynamic_deephit_dataset['time_to_event'][traj_mask]
        
        # Rebuild index_ptr
        new_index_ptr = np.zeros(len(indices_to_keep) + 1, dtype=np.int32)
        for i, idx in enumerate(indices_to_keep):
            start_idx = self.dynamic_deephit_dataset['index_ptr'][idx]
            end_idx = self.dynamic_deephit_dataset['index_ptr'][idx + 1]
            new_index_ptr[i + 1] = new_index_ptr[i] + (end_idx - start_idx)
        
        self.dynamic_deephit_dataset['index_ptr'] = new_index_ptr
        # logger.info(f"Number of timesteps after filtering to cohort {cohort}: {self.dynamic_deephit_dataset['trajectories'].shape[0]}")
        logger.info(f"Number of timesteps after filtering by stay ids: {self.dynamic_deephit_dataset['trajectories'].shape[0]}")


    
    def filter_cohort(self, config_manager):
        """
        Filters the dynamic_deephit_dataset to only include the specified cohort.
        """
        cohort = config_manager._data_config['cohort']
        if cohort is None or cohort == 'all':
            return
        
        # load cohort stay_ids from csv
        stay_ids_path = config_manager.data_config['cohort_stay_ids_path']
        cohort_df = pd.read_csv(stay_ids_path)
        cohort_stay_ids = set(cohort_df['stay_id'].unique().tolist())
        # sort cohort_stay_ids
        cohort_stay_ids = sorted(cohort_stay_ids)
        logger.info(f"Number of stays in cohort {cohort}: {len(cohort_stay_ids)}")
        logger.info(f"Filtering dynamic_deephit_dataset to only include these stay_ids.")

        # # filter the dynamic_deephit_dataset
        # stay_ids = self.dynamic_deephit_dataset['stay_ids']
        # logger.info(f"Original number of stays in dynamic_deephit_dataset: {len(stay_ids)}")
        # logger.info(f"Original number of timesteps in dynamic_deephit_dataset: {self.dynamic_deephit_dataset['trajectories'].shape[0]}")

        # # Create boolean mask for stays to keep
        # mask = np.isin(stay_ids, list(cohort_stay_ids))
        # indices_to_keep = np.where(mask)[0]
        # logger.info(f"Number of stays after filtering to cohort {cohort}: {len(indices_to_keep)}")

        # # Build trajectory mask
        # traj_mask = np.zeros(self.dynamic_deephit_dataset['trajectories'].shape[0], dtype=bool)
        # for idx in indices_to_keep:
        #     start_idx = self.dynamic_deephit_dataset['index_ptr'][idx]
        #     end_idx = self.dynamic_deephit_dataset['index_ptr'][idx + 1]
        #     traj_mask[start_idx:end_idx] = True

        # # Apply masks to all arrays at once
        # self.dynamic_deephit_dataset['stay_ids'] = self.dynamic_deephit_dataset['stay_ids'][mask]
        # self.dynamic_deephit_dataset['event'] = self.dynamic_deephit_dataset['event'][mask]
        # self.dynamic_deephit_dataset['trajectories'] = self.dynamic_deephit_dataset['trajectories'][traj_mask]
        # self.dynamic_deephit_dataset['time_to_event_raw'] = self.dynamic_deephit_dataset['time_to_event_raw'][traj_mask]
        # self.dynamic_deephit_dataset['time_to_event'] = self.dynamic_deephit_dataset['time_to_event'][traj_mask]
        
        # # Rebuild index_ptr
        # new_index_ptr = np.zeros(len(indices_to_keep) + 1, dtype=np.int32)
        # for i, idx in enumerate(indices_to_keep):
        #     start_idx = self.dynamic_deephit_dataset['index_ptr'][idx]
        #     end_idx = self.dynamic_deephit_dataset['index_ptr'][idx + 1]
        #     new_index_ptr[i + 1] = new_index_ptr[i] + (end_idx - start_idx)
        
        # self.dynamic_deephit_dataset['index_ptr'] = new_index_ptr
        # logger.info(f"Number of timesteps after filtering to cohort {cohort}: {self.dynamic_deephit_dataset['trajectories'].shape[0]}")

        self.filter_by_stay_ids(cohort_stay_ids, config_manager)    
    
    
    def add_to_dynamic_deephit_dataset_h5(self, file_path, key):
        # get dimensionality of data to be saved
        data = self.dynamic_deephit_dataset[key]
        dimensions = data.shape
        chunks = (1, ) + dimensions[1:]
                
        with h5py.File(file_path, 'a') as f:
            f.create_dataset(key, data=self.dynamic_deephit_dataset[key], dtype=np.float32, 
                chunks=chunks)
            

    def drop_long_trajectories(self, trajectories, stay_ids, index_ptr):
        """
        Drops trajectories longer than max_time from the given trajectories, stay_ids, and index_ptr.
        Returns the updated trajectories, stay_ids, and index_ptr.
        """
        trajectory_lengths = index_ptr[1:] - index_ptr[:-1]

        trajectories_to_remove = np.where(trajectory_lengths > self.max_time)[0]
        stay_ids_to_remove = stay_ids[trajectories_to_remove]
                
        for stay_id in stay_ids_to_remove:
            idx = np.where(stay_ids == stay_id)[0][0]
            start_idx = index_ptr[idx]
            end_idx = index_ptr[idx + 1]

            trajectories = np.delete(trajectories, np.s_[start_idx:end_idx], axis=0)
            stay_ids = np.delete(stay_ids, idx, axis=0)
            index_ptr = np.delete(index_ptr, idx + 1, axis=0)
            # update index_ptr
            index_ptr[idx + 1:] -= (end_idx - start_idx)
        
        return trajectories, stay_ids, index_ptr

    def add_time_column(self, trajectories, stay_ids, index_ptr, feature_names):
        n_samples = trajectories.shape[0]
        time_col = np.zeros((n_samples, 1))
        for j in range(len(stay_ids)):
            start_idx = index_ptr[j]
            end_idx = index_ptr[j+1]
            length = end_idx - start_idx
            time_values = np.arange(1, length + 1).reshape(-1, 1)
            time_col[start_idx:end_idx, 0] = time_values[:, 0]
        # append time_col to trajectories
        trajectories = np.hstack((time_col, trajectories))
        return trajectories, ['time'] + feature_names

    
    def normalize(self, config_manager=config_manager):

        if self.norm_params is not None:
            # if normalization parameters are provided, use them to normalize the data
            self.obs_norm_dict = self.norm_params['obs_norm_dict']
            self.action_minimum = self.norm_params['action_minimum']
            self.action_maximum = self.norm_params['action_maximum']
            self.action_transformers = self.norm_params['action_transformers']

        else:
            self.obs_norm_dict = None
            self.action_minimum = None
            self.action_maximum = None
            self.action_transformers = None

            for i, shard_file in enumerate(self.raw_shard_files):
                logger.info(f"Normalizing shard file: {shard_file}")
                # load the shard file
                with h5py.File(shard_file, 'r') as f:      
                    # convert feature_names from bytes to strings
                    feature_names = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]                
                    stay_ids = f['stay_ids'][:]
                    index_ptr = f['index_ptr'][:]
                    trajectories = f['trajectories'][:]             
                    
                # drop trajectories longer than max_time
                # remove long trajectories beyond max_time
                trajectories, stay_ids, index_ptr = self.drop_long_trajectories(
                    trajectories, stay_ids, index_ptr)
                
                if config_manager.data_config['add_time_col']:
                    # add time column to trajectories
                    trajectories, feature_names = self.add_time_column(trajectories, stay_ids, index_ptr, feature_names)
                
                # get the index of the death feature, and drop it from the trajectories
                death_feature_index = feature_names.index('death')
                # update feature_names
                feature_names.pop(death_feature_index)
                trajectories = np.delete(trajectories, death_feature_index, axis=1)
                
                if i == 0:
                    n_features = len(feature_names)

                    sums = np.zeros(n_features)
                    sq_sums = np.zeros(n_features)
                    counts = 0
                    minima = np.full(n_features, np.inf)
                    maxima = np.full(n_features, -np.inf)

                # compute sums, squared sums, counts, minima, maxima for 1 feature at a time to avoid floating point catastophic cancellation
                for j in range(n_features):
                    feature_trajectories = trajectories[:, j]
                    sums[j] += np.sum(feature_trajectories)
                    sq_sums[j] += np.sum(feature_trajectories ** 2)
                    minima[j] = np.minimum(minima[j], np.min(feature_trajectories))
                    maxima[j] = np.maximum(maxima[j], np.max(feature_trajectories))
                counts += trajectories.shape[0]

            means = sums / counts
            variances = (sq_sums / counts) - (means ** 2)
            # elementwise clamp variances at 0 to avoid tiny negative values from numerical precision
            stds = np.sqrt(np.maximum(variances, 0.0))  # avoid negative variance due to numerical issues

            self.obs_norm_dict = {}
            for i, feature in enumerate(feature_names):
                self.obs_norm_dict[feature] = {
                    'min': minima[i],
                    'max': maxima[i],
                    'mean': means[i],
                    'std': stds[i]
                }

        # now normalize each shard file and save to new file
        for i, shard_file in enumerate(self.raw_shard_files):
            logger.info(f"Normalizing shard file: {shard_file}")
            # load the shard file
            with h5py.File(shard_file, 'r') as f:      
                # convert feature_names from bytes to strings
                feature_names = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]                
                stay_ids = f['stay_ids'][:]
                index_ptr = f['index_ptr'][:]
                trajectories = f['trajectories'][:]

            # drop trajectories longer than max_time
            # remove long trajectories beyond max_time
            trajectories, stay_ids, index_ptr = self.drop_long_trajectories(
                trajectories, stay_ids, index_ptr)
            
            if config_manager.data_config['add_time_col']:
                # add time column to trajectories
                trajectories, feature_names = self.add_time_column(trajectories, stay_ids, index_ptr, feature_names)
            

            # get the index of the death feature, and drop it from the trajectories
            death_feature_index = feature_names.index('death')
            
            # extract death feature values before removing
            death_values = trajectories[:, death_feature_index]
            
            # update feature_names
            feature_names.pop(death_feature_index)
            n_features = len(feature_names)
            trajectories = np.delete(trajectories, death_feature_index, axis=1)

            # normalize the trajectories
            for j, feature in enumerate(feature_names):
                if feature == 'time':
                    data_type = 'continuous'
                    feature_set = 'time'
                else:
                    if feature in self.features_to_use['feature'].values:
                        feature_row = self.features_to_use[self.features_to_use['feature'] == feature]
                        data_type = feature_row['data_type'].values[0]
                        feature_set = feature_row['feature_set'].values[0]
                    else: 
                        # must either be mask or delta
                        feature_set = 'blank' # we just need a placeholder

                        if "mask" in feature:
                            data_type = 'binary' # if it's a mask, it's binary, no need to normalize
                        elif "delta" in feature: 
                            data_type = 'delta' # if it's delta, gemini said min-max it
                            

                if data_type == 'ordinal' or data_type == 'delta' or feature_set == 'medications':
                    # min-max normalize the feature
                    max_min_diff = self.obs_norm_dict[feature]['max'] - self.obs_norm_dict[feature]['min']
                    trajectories[:, j] = (trajectories[:, j] - self.obs_norm_dict[feature]['min']) / max_min_diff  
                
                elif data_type == 'continuous':
                    # z-score normalize the feature
                    trajectories[:, j] = (trajectories[:, j] - self.obs_norm_dict[feature]['mean']) / self.obs_norm_dict[feature]['std']
                
                # for binary features, do nothing
                # throw error if any nans, infs, or valeus greater than 1e6
                if np.isnan(trajectories[:, j]).any():
                    raise ValueError(f"NaNs found in normalized feature {feature} in shard {shard_file}")
                if np.isinf(trajectories[:, j]).any():
                    raise ValueError(f"Infs found in normalized feature {feature} in shard {shard_file}")
                if np.abs(trajectories[:, j]).max() > 1e6:
                    raise ValueError(f"Extreme values found in normalized feature {feature} in shard {shard_file}")
                
            
            normalized_shard_file = shard_file.replace(".h5", "_normalized.h5")

            with h5py.File(normalized_shard_file, 'w') as f:
                f.create_dataset('feature_names', data=[name.encode('utf-8') for name in feature_names])
                f.create_dataset('stay_ids', data=stay_ids, dtype=np.int32)
                f.create_dataset('index_ptr', data=index_ptr, dtype=np.int32)
                f.create_dataset('trajectories', data=trajectories.astype(np.float32), dtype=np.float32, 
                                 chunks=(1, trajectories.shape[1]))
                f.create_dataset('event', data=death_values.astype(np.int32), dtype=np.int32)

            logger.info(f"Saved normalized shard file: {normalized_shard_file}")   

    def get_normalization_params(self):

        # check if self has the "obs_norm_dict" attribute
        if not hasattr(self, 'obs_norm_dict'):
            return None

        return {
            "obs_norm_dict": self.obs_norm_dict,
            "action_minimum": self.action_minimum,
            "action_maximum": self.action_maximum,
            "action_transformers": self.action_transformers
        }   
    

    @ staticmethod
    def load_dynamic_deephit_dataset_from_h5(file_path):
        with h5py.File(file_path, 'r') as f:
            dynamic_deephit_dataset = {}
            dynamic_deephit_dataset['feature_names'] = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]
            dynamic_deephit_dataset['stay_ids'] = f['stay_ids'][:]
            dynamic_deephit_dataset['index_ptr'] = f['index_ptr'][:]
            dynamic_deephit_dataset['trajectories'] = f['trajectories'][:].astype(np.float32)
            dynamic_deephit_dataset['event'] = f['event'][:].astype(np.int32)
            dynamic_deephit_dataset['time_to_event_raw'] = f['time_to_event_raw'][:].astype(np.int32)
            dynamic_deephit_dataset['time_to_event'] = f['time_to_event'][:].astype(np.int32)

            if 'predictions' in f:
                dynamic_deephit_dataset['predictions'] = f['predictions'][:].astype(np.float32)
            if 'entropy' in f:
                dynamic_deephit_dataset['entropy'] = f['entropy'][:].astype(np.float32)
            if 'health_score' in f:
                dynamic_deephit_dataset['health_score'] = f['health_score'][:].astype(np.float32)

        return dynamic_deephit_dataset
    
    @ staticmethod
    def concatenate_dynamic_deephit_datasets(dataset_list, save_dir = None):
        # check that all datasets have the same feature_names
        feature_names = dataset_list[0]['feature_names']
        for dataset in dataset_list:
            if not np.array_equal(dataset['feature_names'], feature_names):
                raise ValueError("All datasets must have the same feature_names to concatenate")

        concatenated_dataset = {}
        concatenated_dataset['feature_names'] = feature_names

        stay_ids_list = []
        index_ptr_list = []
        trajectories_list = []
        event_list = []
        time_to_event_raw_list = []
        time_to_event_list = []
        predictions_list = []
        entropy_list = []
        health_score_list = []

        for i, dataset in enumerate(dataset_list):
            stay_ids_list.append(dataset['stay_ids'])
            
            if i == 0:
                index_ptr_list.append(dataset['index_ptr'])
            else:
                dataset['index_ptr'] = dataset['index_ptr'] + index_ptr_list[-1][-1]  # shift index_ptr by the last value of the previous dataset's index_ptr
                index_ptr_list[i-1] = index_ptr_list[i-1][:-1]  # remove the last value of the previous dataset's index_ptr to avoid duplication
                index_ptr_list.append(dataset['index_ptr'])

            trajectories_list.append(dataset['trajectories'])
            event_list.append(dataset['event'])
            time_to_event_raw_list.append(dataset['time_to_event_raw'])
            time_to_event_list.append(dataset['time_to_event'])
        
            predictions_list.append(dataset['predictions'])
            entropy_list.append(dataset['entropy'])
            health_score_list.append(dataset['health_score'])

        concatenated_dataset['stay_ids'] = np.concatenate(stay_ids_list, axis=0)
        concatenated_dataset['index_ptr'] = np.concatenate(index_ptr_list, axis=0)
        concatenated_dataset['trajectories'] = np.concatenate(trajectories_list, axis=0)
        concatenated_dataset['event'] = np.concatenate(event_list, axis=0)
        concatenated_dataset['time_to_event_raw'] = np.concatenate(time_to_event_raw_list, axis=0)
        concatenated_dataset['time_to_event'] = np.concatenate(time_to_event_list, axis=0)
        concatenated_dataset['predictions'] = np.concatenate(predictions_list, axis=0)
        concatenated_dataset['entropy'] = np.concatenate(entropy_list, axis=0)
        concatenated_dataset['health_score'] = np.concatenate(health_score_list, axis=0)

        if save_dir is not None:
            save_path = Path(save_dir) / "concatenated_dynamic_deephit_dataset.h5"
            
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('feature_names', data=concatenated_dataset['feature_names'])
                f.create_dataset('stay_ids', data=concatenated_dataset['stay_ids'], dtype=np.int32)
                f.create_dataset('index_ptr', data=concatenated_dataset['index_ptr'], dtype=np.int32)
                f.create_dataset('trajectories', data=concatenated_dataset['trajectories'], dtype=np.float32, 
                                 chunks=(1, concatenated_dataset['trajectories'].shape[1]))
                f.create_dataset('event', data=concatenated_dataset['event'], dtype=np.int32)
                f.create_dataset('time_to_event_raw', data=concatenated_dataset['time_to_event_raw'], dtype=np.int32)
                f.create_dataset('time_to_event', data=concatenated_dataset['time_to_event'], dtype=np.int32)

                if 'predictions' in concatenated_dataset:
                    f.create_dataset('predictions', data=concatenated_dataset['predictions'], dtype=np.float32)
                if 'entropy' in concatenated_dataset:
                    f.create_dataset('entropy', data=concatenated_dataset['entropy'], dtype=np.float32)
                if 'health_score' in concatenated_dataset:
                    f.create_dataset('health_score', data=concatenated_dataset['health_score'], dtype=np.float32)


        return concatenated_dataset


        
def main():
    pass

if __name__ == "__main__":
    main()