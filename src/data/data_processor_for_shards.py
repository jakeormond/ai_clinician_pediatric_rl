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
import json
import time
import pyarrow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler

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

        # check if config_manager.training_config exists
        if hasattr(config_manager, 'training_config'):
            if self.split == 'train' and config_manager.training_config['tuning']:
                self.split = 'tuning'
        
        # find the shard dir for the given split
        if config_manager.split_num is None:
            shard_dir = os.path.join(config_manager.paths['data_dir'], 
                config_manager.data_config['shard_dir'])
        else:
            shard_dir = os.path.join(config_manager.paths['data_dir'], 
                config_manager.data_config['shard_dir'], f"split_{config_manager.split_num}")

        # switch to tuning dir if we're in tuning mode
        if hasattr(config_manager, 'training_config'):
            if config_manager.training_config['tuning']:
                self.shard_dir = os.path.join(shard_dir, 'tuning_set')
            else:
                self.shard_dir = os.path.join(shard_dir, 'training_set')
        else:
            self.shard_dir = os.path.join(shard_dir, 'training_set')
        
        # load processed data if it exists, otherwise process raw shard files and save the processed data for future use
        if config_manager.cohort == 'sepsis' and hasattr(config_manager, 'unit') and config_manager.unit == 'PICU':
            self.file_name_prefix = f"{self.split}_sepsis_picu"
            if config_manager.training_config['tuning']:
                self.norm_params_prefix = "tuning_sepsis_picu"
            else:
                self.norm_params_prefix = "train_sepsis_picu"
        else:
            self.file_name_prefix = f"{self.split}"
            if config_manager.training_config['tuning']:
                self.norm_params_prefix = "tuning"
            else:
                self.norm_params_prefix = "train"
        # else:
        #     raise ValueError(f"Invalid cohort and unit combination: {config_manager.cohort}, {config_manager.unit}. Supported combinations are: (None, None) and ('sepsis', 'PICU').")

        final_file_name = f"{self.file_name_prefix}_dynamic_deephit_dataset.h5"
        final_file_path = os.path.join(self.shard_dir, final_file_name)
        
        # load if exists
        if os.path.exists(final_file_path):
            logger.info(f"Loading existing dynamic deephit dataset from {final_file_path}")
            
            # load the file directly into self.dynamic_deephit_dataset
            self.dynamic_deephit_dataset = self.load_dynamic_deephit_dataset_from_h5(final_file_path)
            
            # if we are running RL, load reward
            if config_manager.data_config['model_type'] == 'reinforcement_learning': # and config_manager.data_config['use_reward']:
                # get the parent directory for the supervised model which generates the health score
                # supervised_model_dir = os.path.dirname(config_manager.paths['supervised_final_model_dir'])
                # supervised_model_dir = os.path.join(supervised_model_dir, "five_splits", 
                #                                        f"split_{config_manager.split_num}", "final_model")
                # get the folder inside
                # final_model_dir = os.listdir(supervised_model_dir)[0]

                final_model_dir = config_manager.data_config['reward_path']

                pred_split = self.split
                if self.split == 'tuning':
                    pred_split = 'train' # the tuning set predictions are generated from the model trained on the training set, so we need to load the train_set_predictions.h5 file

                if config_manager.cohort == 'sepsis' and hasattr(config_manager, 'unit') and config_manager.unit == 'PICU':
                    pred_split_prefix = f"{pred_split}_sepsis_picu"
                else:
                    pred_split_prefix = f"{pred_split}"
                
                prediction_file_path = os.path.join(final_model_dir, f"{pred_split_prefix}_predictions.h5")

                self.dynamic_deephit_predictions = self.load_dynamic_deephit_predictions_from_h5(prediction_file_path, 
                                                                        keys_to_exclude=['predictions', 'entropy'])

                # load behavioural cloning latents if using
                if config_manager.data_config['use_behavior_cloning_latents']:
                    self.use_behavior_cloning_latents = True
                    bc_model_file_dir = config_manager.data_config['bc_latents_path']
                    bc_prediction_file_path = os.path.join(bc_model_file_dir, f"{pred_split_prefix}_predictions.h5")
                    self.bc_latents = self.load_dynamic_deephit_predictions_from_h5(bc_prediction_file_path, 
                                                                        keys_to_load=['model_outputs', 'compressed_latents', 'index_ptr', 'stay_ids'])
                else:
                    self.use_behavior_cloning_latents = False    

                if self.split == 'tuning': # we need to filter the predictions to only include the stays in the tuning set, which are a subset of the training set
                    tuning_stay_ids = self.dynamic_deephit_dataset['stay_ids']
                    self.filter_by_stay_ids(tuning_stay_ids, dataset_types = ['dynamic_deephit_predictions', 'bc_latents'])

                if config_manager._data_config['cohort'] is not None and config_manager._data_config['cohort'] != 'all':
                    # if we are filtering to a specific cohort, we also need to filter the predictions to only include that cohort
                    logger.info(f"Filtering dynamic deephit predictions to only include cohort: {config_manager._data_config['cohort']}")
                    self.filter_cohort(config_manager)               
                
                if hasattr(config_manager._data_config, 'unit') and config_manager._data_config['unit'] is not None and config_manager._data_config['unit'] != 'all':
                    # if we are filtering to a specific unit, we need to filter the dataset to only include that unit
                    logger.info(f"Filtering dynamic deephit dataset to only include unit: {config_manager._data_config['unit']}")
                    self.filter_unit(config_manager)    


                return
          
            # load the normalization parameters if they exist from the json file

            norm_params_file = os.path.join(self.shard_dir, f"{self.norm_params_prefix}_norm_params.json")
            if os.path.exists(norm_params_file):
                with open(norm_params_file, 'r') as f:
                    norm_params = json.load(f)
                self.obs_norm_dict = norm_params['obs_norm_dict']
                logger.info(f"Loaded normalization parameters from {norm_params_file}")
            else:
                raise ValueError(f"No normalization parameters file found at {norm_params_file}. This may cause issues if you are normalizing the validation or test set, as it should be normalized with the parameters from the training set. Please check that your training set was normalized and that the normalization parameters were saved correctly.")

            if hasattr(config_manager._data_config, 'cohort') and config_manager._data_config['cohort'] is not None and config_manager._data_config['cohort'] != 'all':
                # if we are filtering to a specific cohort, we need to filter the dataset to only include that cohort
                logger.info(f"Filtering dynamic deephit dataset to only include cohort: {config_manager._data_config['cohort']}")
                self.filter_cohort(config_manager)       

            if hasattr(config_manager._data_config, 'unit') and config_manager._data_config['unit'] is not None and config_manager._data_config['unit'] != 'all':
                # if we are filtering to a specific unit, we need to filter the dataset to only include that unit
                logger.info(f"Filtering dynamic deephit dataset to only include unit: {config_manager._data_config['unit']}")
                self.filter_unit(config_manager)     
            
            return

    
        # prepare for normalization
        self.norm_params = norm_params
        self.window_length = config_manager.training_config['window_length']
        self.time_bins = config_manager.time_bins

        self.max_time = self.time_bins[-1]

        # load features to use from config_manager
        self.features_to_use = pd.read_csv(config_manager.data_config['features_to_use_path'])       

        all_files = os.listdir(self.shard_dir)                

        raw_shard_files = [os.path.join(self.shard_dir, f) for f in all_files if 
                                f.startswith(self.split) and f.endswith(".h5")]
        # remove any files that contain "dynamic_deephit_dataset" in the name, as these are already processed files
        raw_shard_files = [f for f in raw_shard_files if "dynamic_deephit_dataset" not in f]

        if len(raw_shard_files) == 0:
            raise ValueError(f"No shard files found for split {self.split} in directory {self.shard_dir}")
        elif len(raw_shard_files) > 1:
            # I've simplified the code to work from only one file - check v1 and v2 to get an idea how to use multiple shards, but didn't implement robust scaling yet, which requires median and IQR, which is trickier for multiple shards
            raise ValueError(f"Multiple shard files found for split {self.split} in directory {self.shard_dir}. Expected only one. Found: {raw_shard_files}")

        # don't need a list, as there should only be one file, confirmed above
        self.raw_shard_file = raw_shard_files[0]
            
        ########### Normalized Shards ##############
        self.normalize(config_manager=config_manager) # this creates self.dynamic_deephit_dataset
            
        # create the time_to_event array
        time_to_event = np.zeros_like(self.dynamic_deephit_dataset['trajectories'][:,0], dtype=np.int32)
        for j in range(len(self.dynamic_deephit_dataset['stay_ids'])):
            start_idx = self.dynamic_deephit_dataset['index_ptr'][j]
            end_idx = self.dynamic_deephit_dataset['index_ptr'][j+1]
            length = end_idx - start_idx
            tte = np.arange(length, 0, -1)
            time_to_event[start_idx:end_idx] = tte

        self.dynamic_deephit_dataset['time_to_event_raw'] = time_to_event.astype(np.int32)

        # bin time_to_events
        tte_bin = np.digitize(self.dynamic_deephit_dataset['time_to_event_raw'], self.time_bins, right=True) - 1                
        self.dynamic_deephit_dataset['time_to_event'] = tte_bin

        # get physiological feature indices for dynamic deephit model, and feature_dim
        self.dynamic_deephit_dataset['feature_dim'] = self.dynamic_deephit_dataset['trajectories'].shape[1]

        self.dynamic_deephit_dataset['physiological_feature_ind'] = [] # for dynamic deephit model, which tries to predict next physiological features
        self.dynamic_deephit_dataset['physiological_feature_type'] = []
        self.dynamic_deephit_dataset['response_zero_ind'] = [] # for final timestep features that should be zeroed out because they are responses to interventions at that timestep, and we don't want to feed the model the true response values at the final timestep, as that would leak information about the target
        self.dynamic_deephit_dataset['carry_forward_ind'] = [] # for rinal timestep features that should be carry-forwarded because they are responses to interventions at that timestep, and we want to carry forward the last observed value instead of using the true response value at the final timestep, to avoid leaking information about the target, but also to provide a more realistic scenario for the model to learn from, as in real life, we wouldn't have access to the true response values at the final timestep during inference, but we would have access to the last observed values that we can carry forward.
        
        if config_manager.hospital == 'mimic_iv':
            physiological_feature_sets = [
                'hemodynamics', 'labs', 'neurological', 'routine_vital_signs', 'respiratory'
                ]
        elif config_manager.hospital == 'choc': # demographics included because it has weight_kg
            physiological_feature_sets = [
                'airway', 'blood_gas', 'demographics', 'respiratory', 'input_output',
                'labs', 'neuro', 'vitals'
                ]
        else:
            raise ValueError(f"Invalid hospital: {config_manager.hospital}. Supported hospitals are: 'mimic_iv' and 'choc'.")
                
        for i, feature in enumerate(self.dynamic_deephit_dataset['feature_names']):
            if feature == 'time':
                continue
            
            if feature in self.features_to_use['feature'].values:
                feature_row = self.features_to_use[self.features_to_use['feature'] == feature]            
                feature_set = feature_row['feature_set'].values[0]
                data_type = feature_row['data_type'].values[0]
                response_intervention = feature_row['response_intervention'].values[0]
                
                if feature_set in physiological_feature_sets and response_intervention == 'response':
                    self.dynamic_deephit_dataset['physiological_feature_ind'].append(i)
                    self.dynamic_deephit_dataset['physiological_feature_type'].append(data_type)

                elif response_intervention == 'intervention':
                    if feature_set in [
                        'fluids_intake', 'medications', 'infusion', 'input_output', 'other_meds',
                        'sedation', 'vasoactive_inotropic_infusion'
                        ]:
                        self.dynamic_deephit_dataset['response_zero_ind'].append(i)
                    else:
                        self.dynamic_deephit_dataset['carry_forward_ind'].append(i)
                
                    if f"{feature}_mask" in self.dynamic_deephit_dataset['feature_names']:
                        mask_ind = self.dynamic_deephit_dataset['feature_names'].index(f"{feature}_mask")
                        self.dynamic_deephit_dataset['carry_forward_ind'].append(mask_ind)
                    
                    if f"{feature}_delta" in self.dynamic_deephit_dataset['feature_names']:
                        delta_ind = self.dynamic_deephit_dataset['feature_names'].index(f"{feature}_delta")
                        self.dynamic_deephit_dataset['carry_forward_ind'].append(delta_ind)

        
        # save the file
        final_file_name = f"{self.file_name_prefix}_dynamic_deephit_dataset.h5"
        final_file_path = os.path.join(self.shard_dir, final_file_name)

        # save the dynamic deephit dataset to h5 file for future use
        self.feature_dim = self.dynamic_deephit_dataset['trajectories'].shape[1]
        self.dynamic_deephit_dataset_to_h5(final_file_path)
        
        # if config_manager._data_config['cohort'] is not None and config_manager._data_config['cohort'] != 'all':
        #     # filter the dataset to only include the specified cohort
        #     logger.info(f"Filtering dynamic deephit dataset to only include cohort: {config_manager._data_config['cohort']}")
        #     self.filter_cohort(config_manager)

    
    def process_reward(self, config_manager, hyperparams, health_score_median=None):
        self.reward_type = config_manager.reward_type
        index_ptr = self.dynamic_deephit_dataset['index_ptr']
        mortality = self.dynamic_deephit_dataset['event']  # 1 for mortality, 0 for survival

        if self.reward_type == 'health_score':
            """ 
            Health score and hourly change are scaled equally and then combined to get the final reward, 
            which is then scaled again to keep gradients in an ideal range.
            """
            # score is 0 to 100
            health_score = self.dynamic_deephit_predictions['health_score'] 

            # get hourly_health_change            
            # if hyperparams['potential_based_reward_shaping']: # we apply discount to the next state's health score, so we need to shift the hourly health change by one timestep to align with the current state
            #     health_score_pbrs = copy.deepcopy(health_score)
            #     # we zero out the final timestep health score, since this is only used as next state,
            #     # the final timestep is the one before. And with the shifted health score, 
            #     # we also need to zero out the last used timestamp, this handles that
            #     health_score_pbrs[index_ptr[1:]-1] = 0.0

            #     # shift one timestep back to get the next health score for each current state
            #     health_score_pbrs_next = copy.deepcopy(health_score_pbrs)
            #     health_score_pbrs_next = health_score_pbrs_next[1:]
            #     # make final timestep health score 0, since these have no next state
            #     health_score_pbrs_next[index_ptr[1:-1] - 1] = 0.0
            #     # append on a final zero
            #     health_score_pbrs_next = np.append(health_score_pbrs_next, 0.0)
            #     # also make the second to last timestep health score 0, 
            #     # since for pbrs, we have to zero out the final next health state
            #     health_score_pbrs_next[index_ptr[1:]-2] = 0.0
                
            #     # discount the next health score by the discount factor
            #     discount = hyperparams['discount']
            #     discounted_health_score_pbrs_next = health_score_pbrs_next * discount
             
            #     # calculate the potential-based reward shaping term as the difference between the discounted next health score and the current health score
            #     hourly_health_change = discounted_health_score_pbrs_next - health_score_pbrs
            
            #     # apply symmetric scaling -  we want max abs value to be about 2/3 of hyperparams['terminal_health_score_reward']
            #     hourly_health_change = hourly_health_change / np.max(np.abs(hourly_health_change))

            # else:
            hourly_health_change = np.append(np.diff(health_score), 0.) 
            hourly_health_change[index_ptr[:-1]-1] = 0.0  
        
            # clip hourly_health_change to hyperparms['health_score_hourly_change_clip']
            hourly_max = hyperparams['health_score_hourly_change_clip']
            hourly_min = -hyperparams['health_score_hourly_change_clip']
            hourly_health_change = np.clip(hourly_health_change, hourly_min, hourly_max)
                        
            # # min_max the hourly_health_change to [-1, 1]
            # hourly_health_change = ((hourly_health_change - hourly_min) / \
            #     (hourly_max - hourly_min)) * 2 - 1            
            
            # if health_score_median is None, calculate it
            if health_score_median is None:
                health_score_median = np.median(health_score)
            
            # subtract the median from the health score to center it around 0, which can help with training stability
            health_score = health_score - health_score_median
            
            # shift health score back one index, as we want the reward based on next health score
            health_score = health_score[1:]
            # make final timestep health score 0, since these have no next state
            health_score[index_ptr[1:-1] - 1] = 0.0
            # append on a final 0
            health_score = np.append(health_score, 0.0)
            
            # # min_max scale the health score to [-1, 1] !!!!! CHECK IF YOU SHOULD BE USING THE NEXT TIMESTEP !!!!!!!!!!!!!
            # max_health_score = 100.0
            # min_health_score = 0.0
            # health_score = (((health_score - min_health_score) / \
            #     (max_health_score - min_health_score))) * 2 - 1
            
            # raise value error if health_score_scale and hourly_health_change_scale don't sum to 1
            health_score_scale = hyperparams['health_score_scale']
            hourly_change_scale = hyperparams['health_score_hourly_change_scale']            
            if health_score_scale + hourly_change_scale != 1.0:
                raise ValueError("health_score_scale and hourly_health_change_scale should sum to 1, " \
                    f"but got health_score_scale = {health_score_scale} and hourly_health_change_scale = \
                        {hourly_change_scale}. Please adjust the hyperparameters accordingly.")
            # scale by hyperparameters
            health_score *= hyperparams['health_score_scale']
            hourly_health_change *= hyperparams['health_score_hourly_change_scale']

            # add them together to get the final reward
            health_score_total = health_score + hourly_health_change

            # to address the issue that current hourly_health_change based models prefer zero vaso at all timesteps, we can run a positive control by adding vaso_dose based reward
            if hyperparams['add_reward_to_vaso_dosing']:
                vaso_dose_ind = self.dynamic_deephit_dataset['feature_names'].index('vis')
                vaso_dose = self.dynamic_deephit_dataset['trajectories'][:, vaso_dose_ind]
                # scale vaso_dose to [0, 1] by dividing by max value
                vaso_dose = vaso_dose / np.max(vaso_dose)
                # scale by hyperparameter
                vaso_dose *= hyperparams['vaso_reward_coefficient']
                health_score_total += vaso_dose


            # # apply scaling to total
            # if hyperparams['potential_based_reward_shaping']:
            #     # apply symmetric scaling to the total health score
            #     health_score_total = health_score_total / np.max(np.abs(health_score_total))
            
            # else: 
            #     # min max scaling (to [-1, 1])
            #     # health_score_total = (health_score_total - health_score_total.min()) / \
            #     #     (health_score_total.max() - health_score_total.min()) * 2 - 1         
            #     pass   

            # finally apply a scaling factor to keep gradients in the sweet spot
            total_health_score_scale = hyperparams['total_health_score_scale']
            health_score_total *= total_health_score_scale

            # # add terminal rewards if using PBRS
            # if hyperparams['potential_based_reward_shaping']:
            #     terminal_survival_score = hyperparams['terminal_health_score_reward'] * total_health_score_scale
            #     terminal_mortality_score = -hyperparams['terminal_health_score_reward'] * total_health_score_scale

            #     final_indices = index_ptr[1:] - 2
            #     outcome = self.dynamic_deephit_dataset['event']  # 1 for mortality, 0 for survival
                
            #     # apply terminal_survival_score to outcome == 0 and terminal_mortality_score to outcome == 1 at the final indices
            #     health_score_total[final_indices] += np.where(outcome == 1, terminal_mortality_score, terminal_survival_score)

            # make final timestep rewards 0, since these have no next state and don't get used
            health_score_total[index_ptr[1:] - 1] = 0.0
            health_score_total[-1] = 0.0
            self.dynamic_deephit_predictions['reward'] = health_score_total.astype(np.float32)                                         
        
            return health_score_median

        elif self.reward_type == 'terminal':
            # process terminal rewards monte carlo style
            terminal_reward_scale = hyperparams['terminal_reward_scale']
            terminal_discount = hyperparams['terminal_discount']
            
            mc_reward = self.compute_monte_carlo_terminal_rewards(index_ptr, mortality, terminal_reward_scale, terminal_discount)
            self.mc_reward = mc_reward.astype(np.float32)
            median_reward = np.median(self.mc_reward)
            return median_reward

        else: 
            raise ValueError(f"Invalid reward type: {self.reward_type}. Supported reward types are: 'health_score' and 'time_to_event'.")


    def compute_monte_carlo_terminal_rewards(self, index_ptr, mortality, terminal_reward_scale, terminal_discount):
        """
        Computes a discounted Monte Carlo terminal reward for every timestep
        in every trajectory: target_reward[t] = terminal_discount^(steps_remaining) * signed_outcome

        where steps_remaining is the number of steps from t to the trajectory's
        final timestep, and signed_outcome is +terminal_reward_scale for
        discharge (mortality=0) or -terminal_reward_scale for death (mortality=1).

        Parameters
        ----------
        index_ptr : array of shape (n_trajectories + 1,)
            index_ptr[i] is the start index of trajectory i in the flat
            timestep arrays; index_ptr[i+1] is one-past-the-end (standard
            CSR-style pointer convention).
        mortality : array of shape (n_trajectories,)
            1 if trajectory i ends in mortality, 0 if discharge.
        terminal_reward_scale : float
            Magnitude of the terminal reward (sign applied based on mortality).
        terminal_discount : float
            Discount factor used ONLY for this MC terminal-reward computation —
            keep this separate from whatever discount your intermediate
            health-score reward model uses.

        Returns
        -------
        mc_reward : array of shape (n_timesteps,)
            Flat array, same length/ordering as your existing per-timestep
            arrays (state, action, etc.), giving the discounted terminal
            reward target for each timestep.
        """
        n_trajectories = len(index_ptr) - 1
        n_timesteps = index_ptr[-1]
        mc_reward = np.zeros(n_timesteps, dtype=np.float64)

        signed_outcome = np.where(mortality == 1, -terminal_reward_scale, terminal_reward_scale)

        for i in range(n_trajectories):
            start = index_ptr[i]
            end = index_ptr[i + 1]  # one-past-the-end
            traj_len = end - start

            # steps_remaining[t] = distance from timestep t to the FINAL
            # timestep of this trajectory (0 at the last step, traj_len-1 at the first)
            steps_remaining = np.arange(traj_len - 1, -1, -1)

            discounts = terminal_discount ** steps_remaining
            mc_reward[start:end] = discounts * signed_outcome[i]

        return mc_reward   


    def create_rl_dataset(
            self, states="model_outputs", 
            scaler=None, 
            action_names=None, 
            use_compressed_latents=False,
            n_step=None,
            gamma=None, 
            actions_shift=False,
        ):
        self.rl_dataset = {}

        if type(action_names) == list:
            action_ind = []
            for action_name in action_names:
                if action_name in self.dynamic_deephit_dataset['feature_names']:
                    ind = self.dynamic_deephit_dataset['feature_names'].index(action_name)
                    action_ind.append(ind)
                else:
                    raise ValueError(f"Action name {action_name} not found in feature names.") 
        else:
            action_ind = self.dynamic_deephit_dataset['feature_names'].index(action_names)
        
        # get trajectory starts and ends
        all_ind = np.arange(len(self.dynamic_deephit_dataset['trajectories']))
        trajectory_start_inds = self.dynamic_deephit_dataset['index_ptr'][:-1]
        trajectory_end_inds = self.dynamic_deephit_dataset['index_ptr'][1:] - 1

        state_ind = np.setdiff1d(all_ind, trajectory_end_inds)  # we will exclude the end of trajectory timesteps since these have no next state and action labels are ambiguous (since they could be either the action taken at that timestep or the action taken at the previous timestep)
        # next_state_ind = np.setdiff1d(all_ind, trajectory_start_inds)  # we will exclude the start of trajectory timesteps from the next states for the same reason as above
        next_state_ind = state_ind + 1  # we will exclude the start of trajectory timesteps from the next states for the same reason as above

        # get states
        if states == "model_outputs":
            
            # combine the self.dynamic_deephit_predictions["model_outputs"] and self.bc_latents["model_outputs"] by concatenating along the feature dimension, to create a larger state representation that includes both the dynamic deephit predictions and the behavioral cloning latents. We will then scale this combined state representation before feeding it into the RL model.
            if not use_compressed_latents:
                if self.use_behavior_cloning_latents:
                    model_outputs = np.concatenate([self.dynamic_deephit_predictions["model_outputs"], self.bc_latents["model_outputs"]], axis=1)
                else:
                    model_outputs = self.dynamic_deephit_predictions["model_outputs"]
            else:
                if self.use_behavior_cloning_latents:
                    model_outputs = np.concatenate([self.dynamic_deephit_predictions["compressed_latents"], self.bc_latents["compressed_latents"]], axis=1)
                else:
                    model_outputs = self.dynamic_deephit_predictions["compressed_latents"]

            # need to scale the outputs
            if scaler == None:
                # use StandardScaler to scale the model outputs, fit on the states in the training set
                scaler = StandardScaler()
                scaler.fit(model_outputs)
            
            # apply the scaler to the model outputs
            scaled_model_outputs = scaler.transform(model_outputs)                            
            
            self.rl_dataset['states'] = torch.tensor(scaled_model_outputs[state_ind, :], dtype=torch.float32)
            self.rl_dataset['next_states'] = torch.tensor(scaled_model_outputs[next_state_ind, :], dtype=torch.float32)
        else:
            raise NotImplementedError(f"Only states='model_outputs' is currently implemented, but got states={states}")
             
        actions = torch.tensor(self.dynamic_deephit_dataset["trajectories"][:, action_ind], dtype=torch.float32)

        if not actions_shift:
            self.rl_dataset['actions'] = actions[state_ind, :]
            print("using unshifted actions for RL dataset")
        else:
            self.rl_dataset['actions'] = actions[next_state_ind, :]
            print("using shifted actions for RL dataset")

        if self.reward_type == 'health_score':
            print("Using health_score reward type")
            self.rl_dataset['rewards'] = torch.tensor(self.dynamic_deephit_predictions["reward"][state_ind], dtype=torch.float32)
        elif self.reward_type == 'terminal':
            print("Using terminal reward type")
            self.rl_dataset['rewards'] = torch.tensor(self.mc_reward[state_ind+1], dtype=torch.float32) # we use +1 because we want last state to be full reward
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}. Supported reward types are: 'health_score' and 'terminal'.")
      
        # dones = torch.tensor(np.zeros_like(self.dynamic_deephit_predictions["reward"]), dtype=torch.float32)
        dones = torch.tensor(np.zeros_like(model_outputs[:, 0]), dtype=torch.float32)
        dones[trajectory_end_inds] = 1.0  # mark end of trajectories
        self.rl_dataset['dones'] = dones[next_state_ind]  # only keep dones for the next states, since the current states will never be terminal in our setting (since we exclude end of trajectory timesteps)
        
        
        if not actions_shift:
            self.rl_dataset['next_actions'] = actions[next_state_ind, :]
        elif actions_shift:
            next_next_state_ind = next_state_ind + 1
            # drop final index
            next_next_state_ind = next_next_state_ind[:-1]
            # get next_next_actions
            next_next_actions = actions[next_next_state_ind, :]
            # concatenate on the final index of next_actions with zeros, since there is no next action for the final state
            next_actions_final = torch.zeros_like(next_next_actions[0, :], dtype=torch.float32)
            next_next_actions = torch.cat([next_next_actions, next_actions_final.unsqueeze(0)], dim=0)
            # get done indices from self.rl_dataset['dones'] 
            done_inds = torch.where(self.rl_dataset['dones'] == 1)[0]
            # set all done_inds to next_actions_final
            next_next_actions[done_inds, :] = next_actions_final.unsqueeze(0)
            # store in rl_dataset
            self.rl_dataset['next_actions'] = next_next_actions

        mortality = self.dynamic_deephit_dataset['event']
        mortality_full = np.zeros_like(model_outputs[:, 0], dtype=np.float32)
        # make full mortality trajectoires == 1
        mortality_start_inds = trajectory_start_inds[mortality == 1]
        mortality_end_inds = trajectory_end_inds[mortality == 1]

        for start, end in zip(mortality_start_inds, mortality_end_inds):
            mortality_full[start:end+1] = 1.0

        #  # set the mortality value at the start of each trajectory to the mortality value for that trajectory
        mortality_full = mortality_full[next_state_ind]  # only keep mortality for the next states, since the current states will never be terminal in our setting (since we exclude end of trajectory timesteps)
        self.rl_dataset['mortality'] = torch.tensor(mortality_full, dtype=torch.float32)

        if n_step <= 1:
            return scaler


        #################### NOT CURRENTLY USED #####################
        # After computing states, actions, rewards, dones — add n-step computation
        raw_rewards = self.dynamic_deephit_predictions["reward"]  # full array before indexing

        # Precompute n-step returns per trajectory
        nstep_rewards      = np.zeros_like(raw_rewards, dtype=np.float32)
        nstep_bootstrap_ind = np.arange(len(raw_rewards), dtype=np.int64)  # default: bootstrap from t+1
        nstep_dones        = np.zeros_like(raw_rewards, dtype=np.float32)
    
        for traj_idx, (start, end) in enumerate(zip(trajectory_start_inds, trajectory_end_inds)):
            traj_rewards = raw_rewards[start:end+1]  # (T,)
            T = len(traj_rewards)

            for t in range(T):
                G        = 0.0
                discount = 1.0
                terminated = False

                for k in range(n_step):
                    if t + k >= T:
                        terminated = True
                        break
                    G        += discount * traj_rewards[t + k]
                    discount *= gamma

                nstep_rewards[start + t] = G

                if not terminated and t + n_step < T:
                    # Bootstrap from t+n_step (absolute index)
                    nstep_bootstrap_ind[start + t] = start + t + n_step
                    nstep_dones[start + t]         = 0
                else:
                    # Trajectory ended within window
                    nstep_bootstrap_ind[start + t] = end  # doesn't matter, masked by done
                    nstep_dones[start + t]         = 1

        # Index into these using state_ind (same as everything else)
        self.rl_dataset['nstep_rewards']       = torch.tensor(nstep_rewards[state_ind], dtype=torch.float32)
        self.rl_dataset['nstep_next_states']   = torch.tensor(scaled_model_outputs[nstep_bootstrap_ind[state_ind]], dtype=torch.float32)
        self.rl_dataset['nstep_next_actions']  = actions[nstep_bootstrap_ind[state_ind]]
        self.rl_dataset['nstep_dones']         = torch.tensor(nstep_dones[state_ind], dtype=torch.float32)
        self.rl_dataset['gamma_n']             = gamma ** n_step  # store γⁿ for target computation

        return scaler


    def get_start_states(self):
        dones = self.rl_dataset['dones'].numpy()
        final_ind = np.where(dones == 1)[0]
        start_ind = np.concatenate([[0], final_ind[:-1] + 1])  # start indices are just after each final index, except for the first one which is 0
        return self.rl_dataset['states'][start_ind]



    def dynamic_deephit_dataset_to_h5(self, file_path, keys_to_exclude=None):
        if keys_to_exclude is None:
            keys_to_exclude = []
        
        with h5py.File(file_path, 'w') as f:
            # any dataset should have at minimum feature_names, stay_ids, index_ptr, 
            # f.create_dataset('feature_names', data=self.dynamic_deephit_dataset['feature_names'])
            f.create_dataset('feature_names', data=np.array([name.encode('utf-8') for name in self.dynamic_deephit_dataset['feature_names']], dtype='S'))
            f.create_dataset('stay_ids', data=self.dynamic_deephit_dataset['stay_ids'], dtype=np.int32)
            f.create_dataset('index_ptr', data=self.dynamic_deephit_dataset['index_ptr'], dtype=np.int32)
            f.create_dataset('response_zero_ind', data=np.array(self.dynamic_deephit_dataset['response_zero_ind'], dtype=np.int32), dtype=np.int32)
            f.create_dataset('carry_forward_ind', data=np.array(self.dynamic_deephit_dataset['carry_forward_ind'], dtype=np.int32), dtype=np.int32)
            
            if 'trajectories' in self.dynamic_deephit_dataset and 'trajectories' not in keys_to_exclude:
                f.create_dataset('trajectories', data=self.dynamic_deephit_dataset['trajectories'], dtype=np.float32, 
                                chunks=(1, self.dynamic_deephit_dataset['trajectories'].shape[1]))
            if 'event' in self.dynamic_deephit_dataset and 'event' not in keys_to_exclude:            
                f.create_dataset('event', data=self.dynamic_deephit_dataset['event'], dtype=np.int32)
            
            if 'time_to_event_raw' in self.dynamic_deephit_dataset and 'time_to_event_raw' not in keys_to_exclude:
                f.create_dataset('time_to_event_raw', data=self.dynamic_deephit_dataset['time_to_event_raw'], dtype=np.int32)
                
            if 'time_to_event' in self.dynamic_deephit_dataset and 'time_to_event' not in keys_to_exclude:
                f.create_dataset('time_to_event', data=self.dynamic_deephit_dataset['time_to_event'], dtype=np.int32)
            
            if 'physiological_feature_ind' in self.dynamic_deephit_dataset and 'physiological_feature_ind' not in keys_to_exclude:
                f.create_dataset('physiological_feature_ind', data=np.array(self.dynamic_deephit_dataset['physiological_feature_ind'], dtype=np.int32), dtype=np.int32)
            if 'physiological_feature_type' in self.dynamic_deephit_dataset and 'physiological_feature_type' not in keys_to_exclude:
                f.create_dataset('physiological_feature_type', data=np.array(self.dynamic_deephit_dataset['physiological_feature_type'], dtype='S20'), dtype='S20')
            if 'feature_dim' in self.dynamic_deephit_dataset and 'feature_dim' not in keys_to_exclude:
                f.create_dataset('feature_dim', data=np.array(self.dynamic_deephit_dataset['feature_dim'], dtype=np.int32), dtype=np.int32)
            
            if 'predictions' in self.dynamic_deephit_dataset:
                f.create_dataset('predictions', data=self.dynamic_deephit_dataset['predictions'], dtype=np.float32)
            if 'targets' in self.dynamic_deephit_dataset:
                f.create_dataset('targets', data=self.dynamic_deephit_dataset['targets'], dtype=np.float32)
            if 'entropy' in self.dynamic_deephit_dataset:
                f.create_dataset('entropy', data=self.dynamic_deephit_dataset['entropy'], dtype=np.float32)
            if 'health_score' in self.dynamic_deephit_dataset:
                f.create_dataset('health_score', data=self.dynamic_deephit_dataset['health_score'], dtype=np.float32)
            if 'model_outputs' in self.dynamic_deephit_dataset:
                f.create_dataset('model_outputs', data=self.dynamic_deephit_dataset['model_outputs'], dtype=np.float32)
            if 'compressed_latents' in self.dynamic_deephit_dataset:
                f.create_dataset('compressed_latents', data=self.dynamic_deephit_dataset['compressed_latents'], dtype=np.float32)


        logger.info(f"Saved dynamic deephit dataset to {file_path}")
    
    
    def filter_by_stay_ids(self, stay_ids_to_keep, dataset_types=None):
        
        if dataset_types is None:
            dataset_types = ['dynamic_deephit_dataset', 'dynamic_deephit_predictions', 'bc_latents']
        
        for dataset_type in dataset_types:
            # check if dataset_type is an attribute of the class
            if not hasattr(self, dataset_type):
                logger.warning(f"Dataset type '{dataset_type}' is not an attribute of the class.")
                continue

            # filter the dynamic_deephit_dataset
            if dataset_type == 'dynamic_deephit_dataset':
                dataset = self.dynamic_deephit_dataset
            elif dataset_type == 'dynamic_deephit_predictions':
                dataset = self.dynamic_deephit_predictions
            elif dataset_type == 'bc_latents':
                dataset = self.bc_latents
            elif dataset_type == 'shard_data':
                dataset = self.shard_data
            else:
                raise ValueError(f"Invalid dataset type: {dataset_type}. Supported dataset types are: 'dynamic_deephit_dataset' and 'dynamic_deephit_predictions'.")

            stay_ids = dataset['stay_ids']
            logger.info(f"Original number of stays in {dataset_type}: {len(stay_ids)}")
            
            # Create boolean mask for stays to keep
            mask = np.isin(stay_ids, list(stay_ids_to_keep))
            indices_to_keep = np.where(mask)[0]
            
            # logger.info(f"Number of stays after filtering to cohort {cohort}: {len(indices_to_keep)}")
            logger.info(f"Number of stays after filtering by stay ids: {len(indices_to_keep)}")

            # Build trajectory mask
            if 'trajectories' in dataset:
                traj_mask = np.zeros(dataset['trajectories'].shape[0], dtype=bool)
            elif 'compressed_latents' in dataset:
                traj_mask = np.zeros(dataset['compressed_latents'].shape[0], dtype=bool)
            elif 'model_outputs' in dataset:
                traj_mask = np.zeros(dataset['model_outputs'].shape[0], dtype=bool)
            logger.info(f"Original number of timesteps in {dataset_type}: {traj_mask.shape[0]}")

            for idx in indices_to_keep:
                start_idx = dataset['index_ptr'][idx]
                end_idx = dataset['index_ptr'][idx + 1]
                traj_mask[start_idx:end_idx] = True

            # Apply masks to all arrays at once
            dataset['stay_ids'] = dataset['stay_ids'][mask]
            if 'event' in dataset:
                dataset['event'] = dataset['event'][mask]
            if 'trajectories' in dataset:
                dataset['trajectories'] = dataset['trajectories'][traj_mask]
            if 'time_to_event_raw' in dataset:
                dataset['time_to_event_raw'] = dataset['time_to_event_raw'][traj_mask]
            if 'time_to_event' in dataset:
                dataset['time_to_event'] = dataset['time_to_event'][traj_mask]
            if 'model_outputs' in dataset:
                dataset['model_outputs'] = dataset['model_outputs'][traj_mask]
            if 'compressed_latents' in dataset:
                dataset['compressed_latents'] = dataset['compressed_latents'][traj_mask]
            if 'health_score' in dataset:
                dataset['health_score'] = dataset['health_score'][traj_mask]
            
            # Rebuild index_ptr
            new_index_ptr = np.zeros(len(indices_to_keep) + 1, dtype=np.int32)
            for i, idx in enumerate(indices_to_keep):
                start_idx = dataset['index_ptr'][idx]
                end_idx = dataset['index_ptr'][idx + 1]
                new_index_ptr[i + 1] = new_index_ptr[i] + (end_idx - start_idx)
            
            dataset['index_ptr'] = new_index_ptr
            # logger.info(f"Number of timesteps after filtering to cohort {cohort}: {dataset['trajectories'].shape[0]}")
            if 'trajectories' in dataset:
                logger.info(f"Number of timesteps after filtering by stay ids: {dataset['trajectories'].shape[0]}")
            elif 'model_outputs' in dataset:
                logger.info(f"Number of timesteps after filtering by stay ids: {dataset['model_outputs'].shape[0]}")
            elif 'compressed_latents' in dataset:
                logger.info(f"Number of timesteps after filtering by stay ids: {dataset['compressed_latents'].shape[0]}")

            # Explicitly update the instance attribute
            if dataset_type == 'dynamic_deephit_dataset':
                self.dynamic_deephit_dataset = dataset
            elif dataset_type == 'dynamic_deephit_predictions':
                self.dynamic_deephit_predictions = dataset
            elif dataset_type == 'bc_latents':
                self.bc_latents = dataset

        return


    def filter_cohort(self, config_manager, dataset_types=None):
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

        self.filter_by_stay_ids(cohort_stay_ids, dataset_types=dataset_types)    
    

    def filter_unit(self, config_manager, dataset_types=None):
        """
        Filters the dynamic_deephit_dataset to only include the specified unit.
        """
        unit = config_manager._data_config['unit']
        if unit is None or unit == 'all':
            return
        
        # load unit stay_ids from csv
        stay_ids_path = config_manager.data_config['unit_stay_ids_path']
        unit_df = pd.read_csv(stay_ids_path)
        unit_stay_ids = set(unit_df['stay_id'].unique().tolist())
        # sort unit_stay_ids
        unit_stay_ids = sorted(unit_stay_ids)
        logger.info(f"Number of stays in unit {unit}: {len(unit_stay_ids)}")
        logger.info(f"Filtering dataset to only include these stay_ids.")

        self.filter_by_stay_ids(unit_stay_ids, dataset_types=dataset_types)

    
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

    
    def min_max(self, x, norm_dict):
        min_val = norm_dict['min']
        max_val = norm_dict['max']
        if max_val - min_val == 0:
            return np.zeros_like(x)
        scaled_x = (x - min_val) / (max_val - min_val)
        return np.clip(scaled_x, 0.0, 1.0)


    def log_min_max(self, x, norm_dict):
        log_x = np.log1p(x)
        
        log_min = norm_dict['log_min']
        log_max = norm_dict['log_max']
        max_min_diff = log_max - log_min
        
        if max_min_diff == 0:
            return np.zeros_like(log_x)
        
        scaled_x = (log_x - log_min) / max_min_diff

        # clip to [0, 1] to handle OOD in val/test
        return np.clip(scaled_x, 0.0, 1.0)


    def z_score(self, x, norm_dict):
        mean = norm_dict['mean']
        std = norm_dict['std']
        if std == 0:
            return np.zeros_like(x)

        # clip to [5,-5] to handle extreme outliers
        scaled_x = (x - mean) / std
        return np.clip(scaled_x, -5.0, 5.0)

    
    def robust_scale(self, x, norm_dict):
        median = norm_dict['median']
        iqr = norm_dict['iqr']
        if iqr == 0:
            return np.zeros_like(x)

        scaled_x = (x - median) / iqr
        # clip to [-5, 5] to handle extreme outliers
        return np.clip(scaled_x, -5.0, 5.0)


    def normalize(self, config_manager=config_manager):
        # is norm_params already exists, it's because it was generated from the training or tuning set and we're now normalizing val or test
        if self.norm_params is not None:
            self.obs_norm_dict = self.norm_params['obs_norm_dict']

        # if not, we are normalizing the training or tuning set, so we need to compute the normalization parameters from the data
        else:
            self.obs_norm_dict = {}

            logger.info(f"Computing values for normalizing shard file: {self.raw_shard_file}")
            # load the shard file
            self.shard_data = {}
            with h5py.File(self.raw_shard_file, 'r') as f:      
                # convert feature_names from bytes to strings
                self.shard_data['feature_names'] = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]                
                self.shard_data['stay_ids'] = f['stay_ids'][:]
                self.shard_data['index_ptr'] = f['index_ptr'][:]
                self.shard_data['trajectories'] = f['trajectories'][:]             
                
            
            if config_manager._data_config['cohort'] is not None and config_manager._data_config['cohort'] != 'all':
                # if we are filtering to a specific cohort, we need to filter the dataset to only include that cohort
                logger.info(f"Filtering shard file to only include cohort: {config_manager._data_config['cohort']}")
                self.filter_cohort(config_manager, dataset_types = ['shard_data'])       

            if config_manager._data_config['unit'] is not None and config_manager._data_config['unit'] != 'all':
                # if we are filtering to a specific unit, we need to filter the dataset to only include that unit
                logger.info(f"Filtering shard file to only include unit: {config_manager._data_config['unit']}")
                self.filter_unit(config_manager, dataset_types = ['shard_data'])       

            # drop trajectories longer than max_time
            # remove long trajectories beyond max_time
            self.shard_data['trajectories'], self.shard_data['stay_ids'], self.shard_data['index_ptr'] = self.drop_long_trajectories(
                self.shard_data['trajectories'], self.shard_data['stay_ids'], self.shard_data['index_ptr'])
            
            if config_manager.data_config['add_time_col']:
                # add time column to trajectories
                self.shard_data['trajectories'], self.shard_data['feature_names'] = self.add_time_column(
                    self.shard_data['trajectories'], self.shard_data['stay_ids'], self.shard_data['index_ptr'], self.shard_data['feature_names'])

            # get the index of the death feature, and drop it from the trajectories
            death_feature_name = config_manager.data_config['target_col']
            death_feature_index = self.shard_data['feature_names'].index(death_feature_name)
            # update feature_names
            self.shard_data['feature_names'].pop(death_feature_index)
            self.shard_data['trajectories'] = np.delete(self.shard_data['trajectories'], death_feature_index, axis=1)

            # compute sums, squared sums, counts, minima, maxima for 1 feature at a time to avoid floating point catastophic cancellation
            for j, feature in enumerate(self.shard_data['feature_names']):
                self.obs_norm_dict[feature] = {}

                feature_trajectories = self.shard_data['trajectories'][:, j]

                self.obs_norm_dict[feature]['min'] = np.min(feature_trajectories)
                self.obs_norm_dict[feature]['max'] = np.max(feature_trajectories)
                self.obs_norm_dict[feature]['mean'] = np.mean(feature_trajectories)
                self.obs_norm_dict[feature]['std'] = np.std(feature_trajectories)
                self.obs_norm_dict[feature]['median'] = np.median(feature_trajectories)
                self.obs_norm_dict[feature]['iqr'] = np.percentile(feature_trajectories, 75) - np.percentile(feature_trajectories, 25)
                self.obs_norm_dict[feature]['log_min'] = np.min(np.log1p(feature_trajectories))
                self.obs_norm_dict[feature]['log_max'] = np.max(np.log1p(feature_trajectories))

            # save the normalization parameters to a json file for future use when normalizing val and test sets
            norm_params_file = os.path.join(self.shard_dir, f"{self.file_name_prefix}_norm_params.json")
            with open(norm_params_file, 'w') as f:
                json.dump({'obs_norm_dict': self.obs_norm_dict}, f)
            logger.info(f"Saved normalization parameters to {norm_params_file}")
               
        # now normalize the features with either the pre-computed normalization parameters from the training/tuning set, or the ones we just computed if this is the training/tuning set    
        logger.info(f"Normalizing shard file: {self.raw_shard_file}")
        # load the shard file
        self.shard_data = {}
        with h5py.File(self.raw_shard_file, 'r') as f:      
            # convert feature_names from bytes to strings
            self.shard_data['feature_names'] = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]                
            self.shard_data['stay_ids'] = f['stay_ids'][:]
            self.shard_data['index_ptr'] = f['index_ptr'][:]
            self.shard_data['trajectories'] = f['trajectories'][:]

        if config_manager._data_config['cohort'] is not None and config_manager._data_config['cohort'] != 'all':
            # if we are filtering to a specific cohort, we need to filter the dataset to only include that cohort
            logger.info(f"Filtering shard file to only include cohort: {config_manager._data_config['cohort']}")
            self.filter_cohort(config_manager, dataset_types = ['shard_data'])       

        if config_manager._data_config['unit'] is not None and config_manager._data_config['unit'] != 'all':
            # if we are filtering to a specific unit, we need to filter the dataset to only include that unit
            logger.info(f"Filtering shard file to only include unit: {config_manager._data_config['unit']}")
            self.filter_unit(config_manager, dataset_types = ['shard_data'])      

        # drop trajectories longer than max_time
        # remove long trajectories beyond max_time
        self.shard_data['trajectories'], self.shard_data['stay_ids'], self.shard_data['index_ptr'] = self.drop_long_trajectories(
            self.shard_data['trajectories'], self.shard_data['stay_ids'], self.shard_data['index_ptr'])

        if config_manager.data_config['add_time_col']:
            # add time column to trajectories
            self.shard_data['trajectories'], self.shard_data['feature_names'] = self.add_time_column(
                self.shard_data['trajectories'], self.shard_data['stay_ids'], self.shard_data['index_ptr'], self.shard_data['feature_names'])

        # get the index of the death feature
        # death_feature_index = feature_names.index('death')
        death_feature_name = config_manager.data_config['target_col']
        death_feature_index = self.shard_data['feature_names'].index(death_feature_name)

        # extract death feature values before removing
        death_values = self.shard_data['trajectories'][:, death_feature_index]
        
        # drop death and update feature_names
        self.shard_data['feature_names'].pop(death_feature_index)
        self.shard_data['trajectories'] = np.delete(self.shard_data['trajectories'], death_feature_index, axis=1)

        # normalize the trajectories
        # Note: the normalizatio_type column of features_to_use.csv specifies the type of normalization
        # to apply to each feature, except for time, which we have just created, and the delta and mask features,
        # which are normalized in a consistent way
        for j, feature in enumerate(self.shard_data['feature_names']):   
            # TIME - log transform then min-max normalize using log_min and log_max
            if feature == 'time':
                # we will log transform then min max it - note we need to add a window time feature when passing windows into the model.
                self.shard_data['trajectories'][:, j] = self.log_min_max(self.shard_data['trajectories'][:, j], self.obs_norm_dict[feature])  
            
            else:
                if feature in self.features_to_use['feature'].values:
                    feature_row = self.features_to_use[self.features_to_use['feature'] == feature]
                    normalization_type = feature_row['normalization_type'].values[0]
                    
                else: 
                    if "mask" in feature:
                        normalization_type = 'none' # if it's a mask, it's binary, no need to normalize
                    elif "delta" in feature: 
                        normalization_type = 'min_max' # if it's delta, gemini said min-max it
                    else:
                        raise ValueError(f"Feature {feature} not found in features_to_use.csv, and it's not a mask or delta feature. Please check your features_to_use.csv file.")

                if normalization_type == 'none':
                    continue
                elif normalization_type == 'min_max':
                    self.shard_data['trajectories'][:, j] = self.min_max(self.shard_data['trajectories'][:, j], self.obs_norm_dict[feature])  
                elif normalization_type == 'log_min_max':
                    self.shard_data['trajectories'][:, j] = self.log_min_max(self.shard_data['trajectories'][:, j], self.obs_norm_dict[feature])
                elif normalization_type == 'z_score':
                    self.shard_data['trajectories'][:, j] = self.z_score(self.shard_data['trajectories'][:, j], self.obs_norm_dict[feature])
                elif normalization_type == 'robust_scaling':
                    self.shard_data['trajectories'][:, j] = self.robust_scale(self.shard_data['trajectories'][:, j], self.obs_norm_dict[feature])
                else:
                    raise ValueError(f"Invalid normalization type {normalization_type} for feature {feature}. Please check your features_to_use.csv file.")
               
            # for binary features, do nothing
            # throw error if any nans, infs, or valeus greater than 1e6
            if np.isnan(self.shard_data['trajectories'][:, j]).any():
                raise ValueError(f"NaNs found in normalized feature {feature} in shard {self.raw_shard_file}")
            if np.isinf(self.shard_data['trajectories'][:, j]).any():
                raise ValueError(f"Infs found in normalized feature {feature} in shard {self.raw_shard_file}")
            if np.abs(self.shard_data['trajectories'][:, j]).max() > 1e6:
                raise ValueError(f"Extreme values found in normalized feature {feature} in shard {self.raw_shard_file}")
           
        self.dynamic_deephit_dataset = {}
        self.dynamic_deephit_dataset['feature_names'] = self.shard_data['feature_names']
        self.dynamic_deephit_dataset['stay_ids'] = self.shard_data['stay_ids']
        self.dynamic_deephit_dataset['index_ptr'] = self.shard_data['index_ptr']
        self.dynamic_deephit_dataset['trajectories'] = self.shard_data['trajectories'].astype(np.float32)
        
        final_indices = self.shard_data['index_ptr'][1:] - 1    
        final_events = death_values[final_indices].astype(int)
                
        self.dynamic_deephit_dataset['event'] = final_events.astype(np.int32)

        logger.info(f"Finished normalizing shard file: {self.raw_shard_file}")

    
    def get_normalization_params(self):

        # check if self has the "obs_norm_dict" attribute
        if not hasattr(self, 'obs_norm_dict'):
            return None

        return {
            "obs_norm_dict": self.obs_norm_dict
        }   
    

    @ staticmethod
    def load_dynamic_deephit_dataset_from_h5(file_path, keys_to_load=None, keys_to_exclude=None):
        dynamic_deephit_dataset = {}

        if keys_to_load is None:
            keys_to_load = ['stay_ids', 'trajectories', 'index_ptr', 'feature_names', 'physiological_feature_ind', 
                            'physiological_feature_type', 'feature_dim', 'response_zero_ind', 'carry_forward_ind',
                            'event', 'time_to_event_raw', 'time_to_event']
        if keys_to_exclude is not None:
            keys_to_load = [key for key in keys_to_load if key not in keys_to_exclude]

        with h5py.File(file_path, 'r') as f:
            

            dynamic_deephit_dataset['stay_ids'] = f['stay_ids'][:]
            dynamic_deephit_dataset['trajectories'] = f['trajectories'][:].astype(np.float32)
            dynamic_deephit_dataset['index_ptr'] = f['index_ptr'][:]
                        
            dynamic_deephit_dataset['feature_names'] = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]
            dynamic_deephit_dataset['physiological_feature_ind'] = f['physiological_feature_ind'][:].astype(np.int32).tolist()
            dynamic_deephit_dataset['physiological_feature_type'] = [f['physiological_feature_type'][i].decode('utf-8') for i in range(len(f['physiological_feature_type']))]
            dynamic_deephit_dataset['feature_dim'] = f['feature_dim'][()]
            dynamic_deephit_dataset['response_zero_ind'] = f['response_zero_ind'][:].astype(np.int32).tolist()
            dynamic_deephit_dataset['carry_forward_ind'] = f['carry_forward_ind'][:].astype(np.int32).tolist()
            
            dynamic_deephit_dataset['event'] = f['event'][:].astype(np.int32)
            dynamic_deephit_dataset['time_to_event_raw'] = f['time_to_event_raw'][:].astype(np.int32)
            dynamic_deephit_dataset['time_to_event'] = f['time_to_event'][:].astype(np.int32)

        return dynamic_deephit_dataset
    
    @ staticmethod
    def load_dynamic_deephit_predictions_from_h5(file_path, keys_to_load=None, keys_to_exclude=None):
        dynamic_deephit_predictions = {}

        if keys_to_load is None:
            keys_to_load = ['predictions', 'stay_ids', 'index_ptr', 'model_outputs', 'health_score', 'entropy', 'carry_forward_ind', 'response_zero_ind', "compressed_latents"]
        if keys_to_exclude is not None:
            keys_to_load = [key for key in keys_to_load if key not in keys_to_exclude]

        with h5py.File(file_path, 'r') as f:
            if 'event' in keys_to_load:
                dynamic_deephit_predictions['event'] = f['event'][:].astype(np.int32)
            if 'predictions' in keys_to_load:
                dynamic_deephit_predictions['predictions'] = f['predictions'][:].astype(np.float32)
            if 'targets' in keys_to_load:
                dynamic_deephit_predictions['targets'] = f['targets'][:].astype(np.float32)
            if 'stay_ids' in keys_to_load:
                dynamic_deephit_predictions['stay_ids'] = f['stay_ids'][:]
            if 'index_ptr' in keys_to_load:
                dynamic_deephit_predictions['index_ptr'] = f['index_ptr'][:]
            if 'model_outputs' in keys_to_load:
                dynamic_deephit_predictions['model_outputs'] = f['model_outputs'][:].astype(np.float32)
            if 'compressed_latents' in keys_to_load:
                dynamic_deephit_predictions['compressed_latents'] = f['compressed_latents'][:].astype(np.float32)
            if 'health_score' in keys_to_load:    
                dynamic_deephit_predictions['health_score'] = f['health_score'][:].astype(np.float32)
            if 'entropy' in keys_to_load:  
                dynamic_deephit_predictions['entropy'] = f['entropy'][:].astype(np.float32)
            if 'carry_forward_ind' in keys_to_load:   
                dynamic_deephit_predictions['carry_forward_ind'] = f['carry_forward_ind'][:].astype(np.int32)
            if 'response_zero_ind' in keys_to_load:    
                dynamic_deephit_predictions['response_zero_ind'] = f['response_zero_ind'][:].astype(np.int32)


        
        return dynamic_deephit_predictions
    
   
   

        
def main():
    pass

if __name__ == "__main__":
    main()