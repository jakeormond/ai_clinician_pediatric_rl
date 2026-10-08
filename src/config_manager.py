import os
import yaml
import time
import shutil
from pathlib import Path
import pickle
from .setup_environment import load_paths

class ConfigManager:
    def __init__(
            self, 
            args
        ):

        self.config_dir = Path(__file__).parent.parent / "config"
        
        # add all key value pairs from args to self
        for key, value in vars(args).items():
            setattr(self, key, value)

        if not hasattr(args, "experiment_variant"):
            self.experiment_variant = None

        if args.seeds is None or len(args.seeds) == 1:
            self.seed_mode = "single_seed"
            if len(args.seeds) == 1:
                self.seed = args.seeds[0]
        else:
            self.seed_mode = 'five_seeds'

        if self.experiment_type != "supervised_learning":
            self.prediction_type = None  # prediction type is for supervised learning only
 
        self.paths, self.environment = load_paths(self.hospital)
        self.set_data()
        
        if self.mode == "single_fold":        
            
            self.set_experiment_name()
            self._create_experiment_dir()    
            self.set_cv_config()                   
            self._build_wandb_config()
            self.add_fold(self.fold)

        elif self.mode == "single_split":
            if self.experiment_variant == "prediction":
                # find model directory
                if self.model_type == "dynamic_deephit":
                    results_dir = os.path.join(
                        self.paths['final_model_dir'], 
                        "GRU", "final_model"
                    )
                elif self.model_type == "dynamic_deephit_transformer":
                    if self.unit != 'all' and self.cohort != 'all':
                        results_dir = os.path.join(
                            self.paths['supervised_final_model_dir'], 
                            "five_splits", f"split_{self.split_num}", 
                            self.cohort, "intake-vis", f"{self.cohort}_{self.unit.lower()}", "final_model"
                        )
                    elif self.unit == 'PICU' and self.cohort == 'all':
                        results_dir = os.path.join(
                            self.paths['supervised_final_model_dir'], 
                            "five_splits", f"split_{self.split_num}", 
                            "sepsis", "intake-vis", f"{self.unit.lower()}", "final_model"
                        )
                    else:
                        results_dir = os.path.join(
                            self.paths['supervised_final_model_dir'], 
                            "five_splits", f"split_{self.split_num}", 
                            self.cohort, "intake-vis","final_model"
                        )

                elif self.model_type == "dynamic_deephit_transformer" and self.split_num is not None:
                    if self.model_dir is not None:
                        results_dir = os.path.join(
                            self.paths['supervised_final_model_dir'],
                            "five_splits", f"split_{self.split_num}", 
                            self.model_dir, "final_model"
                        )
                    else:                    
                        results_dir = os.path.join(
                            self.paths['supervised_final_model_dir'], 
                            "five_splits", f"split_{self.split_num}", "final_model"
                        )

                elif self.model_type == "bc_transformer":
                    results_dir = os.path.join(
                        self.paths['bc_transformer_final_model_dir'], 
                        self.model_dir, "final_model"
                    )

                elif self.cohort == 'all':
                    # we will build the paths in the prediction function
                    results_dir = None
                    
                else:
                    raise ValueError(f"Unsupported model_type for prediction: {self.model_type}")

                if results_dir is not None:
                    if not os.path.exists(results_dir):
                        self.experiment_dir = Path(results_dir)
                    else:
                        # get the path to the folder inside the results_dir
                        model_folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]
                        if len(model_folders) > 1:
                            raise ValueError("Multiple model folders found in final_model_dir, please specify one")
                        if len(model_folders) > 0:
                            model_folder = model_folders[0]
                            self.experiment_dir = Path(results_dir) / model_folder
                        else:
                            self.experiment_dir = Path(results_dir)


            elif self.experiment_type == "hyperparameter_search_evaluation" or self.experiment_type == "model_evaluation":
                #### THIS NEEDS FIXING!!!!!!!!!!!!! ##############
                
                results_dir = os.path.join(
                    self.paths["rl_models_dir"], self.rl_type,
                    f"split_{self.split_num}", self.cohort, self.actions,
                )
                self.results_dir = results_dir
                    # find the directory within result_dir
                model_folders_final = []
                model_folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]
                for f in model_folders:
                    model_folders_2 = [f2 for f2 in os.listdir(os.path.join(results_dir, f)) if os.path.isdir(os.path.join(results_dir, f, f2))]    
                    for f2 in model_folders_2:
                        model_folders_final.append(os.path.join(f, f2))
                if len(model_folders_final) == 0:
                    raise ValueError("No model folders found in fmodel_dir, please specify one")
                
                self.model_folders = model_folders_final

            elif self.experiment_type == "final_model_evaluation":               
                results_dir = os.path.join(
                        self.paths["rl_models_dir"], self.rl_type,
                        f"split_{self.split_num}", self.cohort, self.actions, "final_model"
                    )
                    # find the directory within result_dir
                model_folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]
                if len(model_folders) > 1:
                    raise ValueError("Multiple model folders found in final_model_dir, please specify one")
                model_folder = model_folders[0]
                self.results_dir = Path(results_dir) / model_folder

            elif self.experiment_variant == "hyperparameter_search_evaluation":
                pass

            else:
                self.create_single_split_dir()

                if self.search_variant == "best_hyperparameters" or \
                    self.search_variant == "best_hyperparameters_fixed_cql_alpha" or \
                    self.search_variant == "single_hyperparameter_sweep":
                    if self.model_type == "bc_with_logprobs":
                        self.best_results_path = os.path.join(
                            self.paths["rl_models_dir"], self.model_type,
                            f"split_{self.split_num}", self.cohort.lower(), self.actions,
                            f"{self.unit.lower()}_patients",
                            "best_results.json"
                        )
                    elif self.model_type == "dynamic_deephit_transformer":
                        self.best_results_path = os.path.join(
                            self.paths["supervised_final_model_dir"], 'five_splits',
                            f"split_{self.split_num}", 'sepsis', self.action_dir,
                            f"{self.unit.lower()}",
                            "best_results.json"
                        )
                    else:
                        if self.reward_cohort == 'all':
                            self.best_results_path = os.path.join(
                                self.paths["rl_models_dir"], self.model_type,
                                f"split_{self.split_num}", self.cohort.lower(), self.actions,
                                f"{self.unit.lower()}_patients_health_score_from_full_dataset",
                                f"best_results_{self.reward_type}.json"
                            )
                        else:
                            if self.reward_cohort == 'restricted':
                                self.best_results_path = os.path.join(
                                    self.paths["rl_models_dir"], self.model_type,
                                    f"split_{self.split_num}", self.cohort.lower(), self.actions,
                                    f"{self.unit.lower()}_patients_health_score_from_restricted_dataset",
                                    f"best_results_{self.reward_type}.json"
                                )
                            elif self.reward_cohort.lower() == 'picu':
                                if self.model_type == "cql_with_logprobs_floor_bound":
                                    self.best_results_path = os.path.join(
                                        self.paths["rl_models_dir"], "cql_with_logprobs",
                                        f"split_{self.split_num}", self.cohort.lower(), self.actions,
                                        f"{self.unit.lower()}_patients_health_score_from_picu_dataset",
                                        "with_actions",
                                        f"best_results_{self.reward_type}.json"
                                    )
                                else:
                                    self.best_results_path = os.path.join(
                                        self.paths["rl_models_dir"], self.model_type,
                                        f"split_{self.split_num}", self.cohort.lower(), self.actions,
                                        f"{self.unit.lower()}_patients_health_score_from_picu_dataset",
                                        "with_actions",
                                        f"best_results_{self.reward_type}.json"
                                    )

        elif self.mode == "analyse_features":
            self.set_experiment_name()
            self._create_experiment_dir()    

        elif self.mode == "aggregate":
            
            if self.experiment_folder is None:
                raise ValueError("experiment_folder must be provided in aggregate mode")
            
            # model type needs to be found from experiment_folder, it is either 
            # xgbclassifier or xgbregressor
            if "xgbclassifier" in self.experiment_folder:
                self.model_type = "xgbclassifier"
            elif "xgbregressor" in self.experiment_folder:
                self.model_type = "xgbregressor"
            else:
                raise ValueError("model_type not found in experiment_folder, must be either xgbclassifier or xgbregressor")
                        
            # join experiment_dir with self.paths['experiments_dir']
            self.experiment_dir = Path(self.paths['experiments_dir']) / self.experiment_folder

            # from inside the experiment_dir, enter the config directory and load cross_validation*.yaml
            config_dir = self.experiment_dir / "config"
            if not config_dir.exists():
                raise FileNotFoundError(f"Config directory does not exist: {config_dir}")
            
            # load the yaml file that begins with "cross_validation" and ends with ".yaml"
            cv_config_files = list(config_dir.glob("cross_validation_*.yaml"))
            if not cv_config_files:
                raise FileNotFoundError("No cross_validation config file found in the experiment directory")
            if len(cv_config_files) > 1:
                raise ValueError("Multiple cross_validation config files found, please specify one")
            
            with open(cv_config_files[0], 'r') as f:
                self.cv_config = yaml.safe_load(f)

        if self.hospital == "choc":
            self.data_config['age_stays_dir'] = os.path.join(
                self.get_data_dir,
                'cohorts',
                'age' 
            )
            self.data_config['age_stay_ids_path'] = os.path.join(
                self.data_config['age_stays_dir'],
                'pediatric_stay_ids.csv'
            )

        if self.cohort is not None and self.cohort != "all":
            self._data_config['cohort'] = self.cohort   
            self.data_config['cohort_dir'] = os.path.join(
                self.get_data_dir,
                'cohorts',
                self._data_config['cohort']
            )
            self.data_config['cohort_stay_ids_path'] = os.path.join(
                self.data_config['cohort_dir'],
                'stay_ids.csv'
            )
        else:
            self._data_config['cohort'] = 'all'

        if hasattr(self, 'unit') and self.unit is not None:
            self._data_config['unit'] = self.unit.lower()
            self.data_config['unit_stay_ids_path'] = os.path.join(
                self.get_data_dir, 'cohorts',
                'icu_units',
                f"{self._data_config['unit'].lower()}_stay_ids.csv"
            )
        else:
            self._data_config['unit'] = None
            self.data_config['unit_stay_ids_path'] = None


    def reset_cohort(self, cohort):
        """Reset the cohort in the data config."""
        self.cohort = cohort
        if self.cohort is not None and self.cohort != "all":
            self._data_config['cohort'] = self.cohort   
            self.data_config['cohort_dir'] = os.path.join(
                self.get_data_dir,
                'cohorts',
                self._data_config['cohort']
            )
            self.data_config['cohort_stay_ids_path'] = os.path.join(
                self.data_config['cohort_dir'],
                'stay_ids.csv'
            )
        else:
            self._data_config['cohort'] = 'all'


    def reset_unit(self,unit):
        """Reset the unit in the data config."""
        self.unit = unit
        if hasattr(self, 'unit') and self.unit is not None:
            self._data_config['unit'] = self.unit.lower()
            self.data_config['unit_stay_ids_path'] = os.path.join(
                self.get_data_dir, 'cohorts',
                'icu_units',
                f"{self._data_config['unit'].lower()}_stay_ids.csv"
            )
        else:
            self._data_config['unit'] = None
            self.data_config['unit_stay_ids_path'] = None


    def reset_age(self):
        """Reset the age in the data config."""
        if self.filter_old:  
            self.data_config['age_dir'] = os.path.join(
                self.get_data_dir,
                'cohorts',
                'age'
            )
            self.data_config['age_stay_ids_path'] = os.path.join(
                self.data_config['age_dir'],
                'pediatric_stay_ids.csv'
            )
        else:
            self._data_config['age'] = None

                
    def get_config_path(self, config_type, custom_path=None):
        """Get config file path with variant support."""
        if custom_path:
            return Path(custom_path)
                
        if config_type == "cv":
            # Try variant-specific first, fall back to default
            variant_name = f"cross_validation_{self.experiment_variant}.yaml"
            variant_path = self.config_dir / "search_spaces" / variant_name
            if variant_path.exists():
                return variant_path
            else:
                raise FileNotFoundError(f"CV config not found for variant: {self.experiment_variant}")

        elif config_type == "training":
            if self.search_variant == "fixed_hyperparameters":
                variant_name = f"{self.model_type}_fixed.yaml"
            else:
                variant_name = f"{self.model_type}.yaml"
            variant_path = self.config_dir / "models" / variant_name
            if variant_path.exists():
                return variant_path
            else:
                raise FileNotFoundError(f"Training config not found for model type: {self.model_type}")

        elif config_type == "search_grid":
            variant_name = f"{self.model_type}_{self.search_variant}.yaml"
            variant_path = self.config_dir / "search_spaces" / variant_name
            if variant_path.exists():
                return variant_path
            else:
                raise FileNotFoundError(f"Search grid config not found for variant: {self.search_variant}")
            
        elif config_type == "wandb":
            # W&B config usually doesn't need variants
            return self.config_dir / "wandb" / "wandb.yaml"
            
        elif config_type == "data":
            if self.experiment_type == "supervised_learning":
                return self.config_dir / "datasets" / f"{self.hospital}_{self.experiment_type}.yaml"
            elif self.experiment_type == "reinforcement_learning" or self.experiment_type == "model_evaluation":
                if self.cohort is None:
                    raise ValueError("Cohort must be specified for reinforcement learning data config")
                # if self.cohort doesn't contain an underscore, then we assume it is just the disease name and we add "_broad" to it
                if '_' in self.cohort:
                    disease_variant = self.cohort[:self.cohort.index('_')]
                else:
                    disease_variant = self.cohort
                
                experiment_type = "reinforcement_learning"
                return self.config_dir / "datasets" / f"{self.hospital}_{disease_variant}_{experiment_type}.yaml"
            
        elif config_type == "rl":
            return self.config_dir / "models" / f"{self.model_type}.yaml"
   

    def set_data(self,):
        data_config_path = self.get_config_path("data")
        if data_config_path is None:
            print("No data config path found, skipping data config loading")
            return
        if not data_config_path.exists():
            raise FileNotFoundError(f"Data config file not found: {data_config_path}")
        with open(data_config_path, 'r') as f:
            self._data_config = yaml.safe_load(f)

        # set self.seed to the seed in the data config
        # self.seed = self._data_config['seed']
    
        self._data_config['prediction_type'] = self.prediction_type

        # if 'dataframe_path' in self._data_config:
        #     if not os.path.isabs(self._data_config['dataframe_path']):
        #         self._data_config['dataframe_path'] = os.path.join(
        #             self.get_data_dir, 
        #             self._data_config['dataframe_path']
        #         )
        #         print(f"Updated dataframe_path to absolute: {self._data_config['dataframe_path']}")
        
        if 'features_to_use_path' in self._data_config:
            if not os.path.isabs(self._data_config['features_to_use_path']):
                self._data_config['features_to_use_path'] = os.path.join(
                    self.get_code_dir, 
                    self._data_config['features_to_use_path']
                )

        # set reward path if reward_type is not "none"
        if self.experiment_type != "supervised_learning": # supervised learning is for generating rewards, so doesn't use them
            if self._data_config['use_reward']:
                # reward_type = self._data_config['reward_type']
                # reward_paths_config_path = self.config_dir / "rewards" / "reward_paths.yaml"
                # reward_path = self.data_config['reward_path']
                reward_path = os.path.join(
                    self.paths['supervised_final_model_dir'],
                    "five_splits",
                )

                split_name = f"split_{self.split_num}"
                
                if self.cohort is not None and self.cohort != "all":
                    split_name = os.path.join(split_name, self.cohort)

                actions = self.data_config.get('actions', None)
                if actions is not None:
                    actions_name = "-".join(actions)
                    split_name = os.path.join(split_name, actions_name)   
                    self.actions = actions_name             
                
                if self.reward_cohort == 'restricted':
                    if self.cohort != 'all' and self.unit != 'all':                    
                        split_name = os.path.join(split_name, f"{self.cohort}_{self.unit.lower()}")
                elif self.reward_cohort == 'picu':
                    split_name = os.path.join(split_name, f"{self.unit.lower()}")

                if not hasattr(self, "include_actions_in_state"):
                    self.include_actions_in_state = "True"

                if self.include_actions_in_state == 'True':
                    final_model_dir = os.path.join(
                        reward_path,
                        split_name,
                        "with_actions"
                    )
                    state_model_dir = final_model_dir
                else:
                    final_model_dir = os.path.join(
                        reward_path,
                        split_name,
                        "without_actions"
                    )
                    state_model_dir = os.path.join(
                        reward_path,
                        split_name,
                        "with_actions"
                    )
                                
                # final_model_dir = os.path.join(
                #     reward_path,
                #     split_name,
                #     "final_model"
                # )
                # find the folder in final_model_dir 

                state_and_reward_model_dirs = {
                    "state_model_dir": state_model_dir,
                    "reward_model_dir": final_model_dir
                }

                for key, path in state_and_reward_model_dirs.items():
                    # no longer using single seed
                    # check if we're single seed or 5 seeds
                    # if self.seeds is not None and len(self.seeds) > 1:
                    #     path = os.path.join(path, "five_seeds")
                    # else:
                    #     path = os.path.join(path, "single_seed")

                    path = os.path.join(path, "five_seeds")

                    # check if there is a folder inside final_model_dir that begins with "choc"
                    inner_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
                    choc_folders = [f for f in inner_folders if f.startswith(self.hospital.lower())]
                    if len(choc_folders) > 1:
                        raise ValueError("Multiple choc folders found in final_model_dir, please specify one")
                    elif len(choc_folders) == 1:
                        path = os.path.join(path, choc_folders[0])
                    else:
                        raise ValueError("No choc folders found in final_model_dir, please specify one")

                    # model_folders = [f for f in os.listdir(final_model_dir) if os.path.isdir(os.path.join(final_model_dir, f))]
                    # if len(model_folders) > 1:
                    #     raise ValueError("Multiple model folders found in reward path, please specify one")
                    # model_folder = model_folders[0]
                    
                    # reward_path = os.path.join(final_model_dir, model_folder) # this is not the final folder
                    if key == "state_model_dir":
                        self._data_config['state_path'] = path
                    elif key == "reward_model_dir":
                        self._data_config['reward_path'] = path 

            if self._data_config['use_behavior_cloning_latents']:
                bc_latents_path = os.path.join(
                    self.paths['bc_transformer_final_model_dir'], 
                    "five_splits"
                )

                split_name = f"split_{self.split_num}"
                if self.cohort is not None and self.cohort != "all":
                    split_name = os.path.join(split_name, self.cohort)
                
                actions = self.data_config.get('actions', None)
                if actions is not None:
                    actions_name = "-".join(actions)
                    split_name = os.path.join(split_name, actions_name)   
                
                final_model_dir = os.path.join(
                    bc_latents_path,
                    split_name,
                    "final_model"
                )

                model_folders = [f for f in os.listdir(final_model_dir) if os.path.isdir(os.path.join(final_model_dir, f))]
                if len(model_folders) > 1:
                    raise ValueError("Multiple model folders found in reward path, please specify one")
                model_folder = model_folders[0]
                bc_path = os.path.join(final_model_dir, model_folder)
                self._data_config['bc_latents_path'] = bc_path


    def _build_wandb_config(self):
        """Build W&B configuration and store it internally."""
        wandb_config_path = self.get_config_path("wandb")
        if not wandb_config_path.exists():
            raise FileNotFoundError(f"W&B config file not found: {wandb_config_path}")
        with open(wandb_config_path, 'r') as f:
            self._wandb_config = yaml.safe_load(f)

        self._wandb_config['hospital'] = self.hospital
        self._wandb_config['experiment_type'] = self.experiment_type
        self._wandb_config['model_type'] = self.model_type        
        self._wandb_config['seed'] = self._data_config['seed'] if self.data_config else None
        self._wandb_config['group'] = f"{self.hospital}_{self.model_type}_{self.experiment_variant}"

        self.wandb_config['experiment_name'] = self.experiment_name       

        # Apply environment-specific config
        if self.environment == 'hpc':
            self._wandb_config['mode'] = 'offline'
        else:
            self._wandb_config['mode'] = 'online'

        # Apply experiment type configuration
        if self.mode == "single_fold":
            self._wandb_config['cv_config'] = self.cv_config
        
        if self.experiment_type == "supervised_learning":
            
            # Build tags
            tags = ["mortality", self.hospital, self.model_type]
            self._wandb_config['tags'] = tags  
   
    def set_cv_config(self):
        """Set cross-validation configuration."""
        cv_config_path = self.get_config_path("cv")
        if not cv_config_path.exists():
            raise FileNotFoundError(f"cv config file not found: {cv_config_path}")      

        with open(cv_config_path, 'r') as f:
            self.cv_config = yaml.safe_load(f)
            

        # get the filename from cv_config_path
        cv_config_filename = cv_config_path.stem
        experiment_copy_path = self.experiment_dir / f"{cv_config_filename}.yaml"
        if not experiment_copy_path.exists():
            shutil.copy2(cv_config_path, experiment_copy_path)
            print(f"Copied CV config to: {experiment_copy_path}")
        else:
            print(f"CV config already exists at: {experiment_copy_path}, skipping copy")

    def load_cv_config(self):
        n_outer_folds = self.cv_config.get('n_outer_folds', None)
        n_inner_folds = self.cv_config.get('n_inner_folds', None)
        target_fold = self.cv_config.get('fold', None)

        if target_fold is None:
            raise ValueError("target_fold must be specified")

        return n_outer_folds, n_inner_folds, target_fold     

    def set_training_config(self):
        if self.search_variant == 'best_hyperparameters' or \
            self.search_variant == 'single_hyperparameter_sweep' or \
            self.search_variant == 'best_hyperparameters_fixed_cql_alpha':
            # load best hyperparameters from f"best_results_{self.reward_type}.json" in the model_dir
            if not os.path.exists(self.best_results_path):
                raise FileNotFoundError(f"Best results file not found: {self.best_results_path}")
            with open(self.best_results_path, 'r') as f:
                best_results = yaml.safe_load(f)
            self.training_config = best_results['hyperparameters']

            numeric_float_keys = [
                "learning_rate", "discount", "tau", "weight_decay",
                "grad_clip", "reward_scaling"
            ]
            numeric_int_keys = ["batch_size", "hidden_dim", "total_steps", "n_step"]

            for k in numeric_float_keys:
                if k in self.training_config and isinstance(self.training_config[k], str):
                    self.training_config[k] = float(self.training_config[k])

            for k in numeric_int_keys:
                if k in self.training_config and isinstance(self.training_config[k], str):
                    self.training_config[k] = int(float(self.training_config[k]))

            # if seed is an attribute of self
            if hasattr(self, 'seed') and self.seed is not None:
                self.training_config['seed'] = self.seed

        else:   
            training_config_path = self.get_config_path("training")
            if not training_config_path.exists():
                raise FileNotFoundError(f"Training config file not found: {training_config_path}")
            with open(training_config_path, 'r') as f:
                self.training_config = yaml.safe_load(f)
            # add the action_mode to the training_config
            if self.model_type != "dynamic_deephit":
                self.training_config['action_mode'] = self.action_mode

        
    def add_fold(self, fold):
        """Add fold information to the config."""
        self._wandb_config['cv_config']['fold'] = fold

    def set_search_grid(self):
        """Add search grid to the W&B config."""
        search_grid_path = self.get_config_path("search_grid")
        if not search_grid_path.exists():
            raise FileNotFoundError(f"Search grid config file not found: {search_grid_path}")
        
        with open(search_grid_path, 'r') as f:
            # Load the search grid config
            self.search_grid = yaml.safe_load(f)        
            self._wandb_config['search_grid'] = self.search_grid

        # get the filename from search_grid_path
        search_grid_filename = search_grid_path.stem
        experiment_copy_path = self.expt_config_dir / f"{search_grid_filename}.yaml"
        if not experiment_copy_path.exists():
            shutil.copy2(search_grid_path, experiment_copy_path)
            print(f"Copied search grid config to: {experiment_copy_path}")
        else:
            print(f"Search grid config already exists at: {experiment_copy_path}, skipping copy")

    def set_experiment_name(self):
        if self.experiment_type == "supervised_learning":
            if self.cohort is not None and self.cohort != "all":
                self.experiment_variant = f"{self.experiment_variant}_{self.cohort}"
            if self.split_num is not None:
                self.experiment_variant = f"{self.experiment_variant}_split_{self.split_num}"
            
            self.experiment_name = \
                f"{self.hospital}_{self.model_type}_{self.prediction_type}_{self.experiment_variant}_{self.start_time}"
        
        elif self.experiment_type == "model_evaluation" or self.experiment_variant == "hyperparameter_search_evaluation":
            return

        elif self.search_variant == "best_hyperparameters" and self.seeds is not None:
            self.experiment_name = \
                f"{self.hospital}_{self.model_type}_{self.actions}_{self.search_variant}_{self.start_time}_with_seeds"
        
        else:
            self.experiment_name = \
                f"{self.hospital}_{self.model_type}_{self.actions}_{self.search_variant}_{self.start_time}"
        # make directory if it doesn't exist
        self._create_experiment_dir()

    def _create_experiment_dir(self):
        """Create and return the base experiment directory path."""       
        self.experiment_dir = Path(self.get_experiments_dir) / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        # create a config dir inside the experiment directory
        # self.expt_config_dir = self.experiment_dir / "config"
        # self.expt_config_dir.mkdir(parents=True, exist_ok=True)

    def set_fold_name(self):
        """Add experiment name to the W&B config."""
        self.fold_name = f"fold_{self.cv_config['fold']}_{self.start_time}"
        self._wandb_config['name'] = self.fold_name        
        self._create_fold_dir(self.fold_name)

    def _create_fold_dir(self, experiment_name):
        """Create and return experiment directory path."""
        fold = self.cv_config.get('fold', None)
        
        # append "fold_{fold}" to experiment_dir name
        new_dir_path = self.experiment_dir.parent / f"{self.experiment_dir.name}_fold_{fold}"
        
        # rename experiment_dir to new_dir_path
        self.experiment_dir.rename(new_dir_path)
        self.experiment_dir = new_dir_path
        

    def create_single_split_dir(self):
        """Create and return single split experiment directory path."""
        self.set_experiment_name()        
        self._create_experiment_dir()
    
    
    
    @property
    def get_experiments_dir(self):
        return self.paths['experiments_dir']
    
    @property
    def get_data_dir(self):
        return self.paths['data_dir']
    
    @property
    def get_code_dir(self):
        return self.paths['code_dir']       
    
    # Properties for accessing hospital config
    @property
    def data_config(self):
        """Get the complete hospital configuration."""
        if self._data_config is None:
            raise ValueError("Hospital not set. Call set_hospital() first.")
        return self._data_config
    
    @property
    def wandb_config(self):
        """Get the W&B configuration."""
        if self._wandb_config is None:
            raise ValueError("W&B config not set. Call _build_wandb_config() first.")
        return self._wandb_config



def load_configuration(config_manager_path):
    if not os.path.exists(config_manager_path):
        print(f"Config manager not found in: {config_manager_path}, skipping evaluation.")
        raise FileNotFoundError(f"Config manager not found in: {config_manager_path}")
    else:
        print(f"Loading config manager from: {config_manager_path}")
        with open(config_manager_path, "rb") as cfg_file:
            config_manager = pickle.load(cfg_file)
    
    # set paths 
    config_manager.paths, config_manager.environment = load_paths(config_manager.hospital)
    if config_manager.environment == "local":
        config_manager.config_dir = Path(config_manager.paths['config_dir'])
    config_manager.set_data()
    config_manager.reset_cohort(config_manager.cohort)
    config_manager.reset_unit(config_manager.unit)
    config_manager.reset_age()

    return config_manager