import os
import yaml
import time
import shutil
from pathlib import Path
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

        if self.experiment_type != "supervised_learning":
            self.prediction_type = None  # prediction type is for supervised learning only
 
        self.paths, self.environment = load_paths()
        self.set_data()

        if self.mode == "single_fold":        
            
            self.set_experiment_name()
            self._create_experiment_dir()    
            self.set_cv_config()                   
            self._build_wandb_config()
            self.add_fold(self.fold)

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
            return self.config_dir / "datasets" / f"{self.hospital}_{self.experiment_type}.yaml"
        
        elif config_type == "rl":
            return self.config_dir / "models" / f"{self.model_type}.yaml"
   

    def set_data(self,):
        data_config_path = self.get_config_path("data")
        if not data_config_path.exists():
            raise FileNotFoundError(f"Data config file not found: {data_config_path}")
        with open(data_config_path, 'r') as f:
            self._data_config = yaml.safe_load(f)

        # set self.seed to the seed in the data config
        self.seed = self._data_config['seed']
    
        self._data_config['prediction_type'] = self.prediction_type

        if 'dataframe_path' in self._data_config:
            if not os.path.isabs(self._data_config['dataframe_path']):
                self._data_config['dataframe_path'] = os.path.join(
                    self.get_data_dir, 
                    self._data_config['dataframe_path']
                )
                print(f"Updated dataframe_path to absolute: {self._data_config['dataframe_path']}")
        
        if 'features_to_use_path' in self._data_config:
            if not os.path.isabs(self._data_config['features_to_use_path']):
                self._data_config['features_to_use_path'] = os.path.join(
                    self.get_code_dir, 
                    self._data_config['features_to_use_path']
                )

        # set reward path if reward_type is not "none"
        if self.experiment_type != "supervised_learning": # supervised learning is for generating rewards, so doesn't use them
            if self._data_config['use_reward']:
                reward_type = self._data_config['reward_type']
                reward_paths_config_path = self.config_dir / "rewards" / "reward_paths.yaml"
                if not reward_paths_config_path.exists():
                    raise FileNotFoundError(f"Reward paths config file not found: {reward_paths_config_path}")
                with open(reward_paths_config_path, 'r') as f:
                    reward_paths = yaml.safe_load(f).get("rewards", {})
                if reward_type:
                    if reward_type not in reward_paths:
                        raise ValueError(f"Reward type '{reward_type}' not found in reward paths config")
                    reward_path = reward_paths[reward_type]
                    if not os.path.isabs(reward_path):
                        reward_path = os.path.join(self.get_experiments_dir, reward_path)
                    self._data_config['reward_path'] = reward_path
                    print(f"Set reward_path to: {self._data_config['reward_path']}")

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
        experiment_copy_path = self.expt_config_dir / f"{cv_config_filename}.yaml"
        if not experiment_copy_path.exists():
            shutil.copy2(cv_config_path, experiment_copy_path)
            print(f"Copied CV config to: {experiment_copy_path}")
        else:
            print(f"CV config already exists at: {experiment_copy_path}, skipping copy")


    def set_training_config(self):
        training_config_path = self.get_config_path("training")
        if not training_config_path.exists():
            raise FileNotFoundError(f"Training config file not found: {training_config_path}")
        with open(training_config_path, 'r') as f:
            self.training_config = yaml.safe_load(f)
        # add the action_mode to the training_config
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
            self.experiment_name = \
                f"{self.hospital}_{self.model_type}_{self.prediction_type}_{self.experiment_variant}_{self.start_time}"
        else:
            self.experiment_name = \
                f"{self.hospital}_{self.model_type}_{self.action_mode}_{self.search_variant}_{self.experiment_variant}_{self.start_time}"
        # make directory if it doesn't exist
        self._create_experiment_dir()

    def _create_experiment_dir(self):
        """Create and return the base experiment directory path."""
        self.experiment_dir = Path(self.get_experiments_dir) / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        # create a config dir inside the experiment directory
        self.expt_config_dir = self.experiment_dir / "config"
        self.expt_config_dir.mkdir(parents=True, exist_ok=True)

    def set_fold_name(self):
        """Add experiment name to the W&B config."""
        self.fold_name = f"fold_{self.cv_config['fold']}_{self.start_time}"
        self._wandb_config['name'] = self.fold_name        
        self._create_fold_dir(self.fold_name)

    def _create_fold_dir(self, experiment_name):
        """Create and return experiment directory path."""
        self.fold_dir = Path(self.get_experiments_dir) / self.experiment_name / self.fold_name
        self.fold_dir.mkdir(parents=True, exist_ok=True)
    
    
    
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