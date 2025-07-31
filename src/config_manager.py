import os
import yaml
from pathlib import Path
from .setup_environment import load_paths

class ConfigManager:
    def __init__(self):
        self.paths, self.environment = load_paths()
        
    def get_experiments_dir(self):
        return self.paths['experiments_dir']
    
    def get_data_dir(self):
        return self.paths['data_dir']
    
    def get_code_dir(self):
        return self.paths['code_dir']
    
    def get_config_path(self, hospital):
        """Get the config path for a specific hospital."""
        return Path(__file__).parent.parent / "config" / f"{hospital}_supervised.yaml"
    
    def load_hospital_config(self, hospital):
        """Load hospital-specific config and update paths."""
        config_path = self.get_config_path(hospital)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update any relative paths to be relative to data_dir
        if 'dataframe_path' in config:
            if not os.path.isabs(config['dataframe_path']):
                config['dataframe_path'] = os.path.join(
                    self.get_data_dir(), 
                    config['dataframe_path']
                )
        
        if 'features_to_use_path' in config:
            if not os.path.isabs(config['features_to_use_path']):
                config['features_to_use_path'] = os.path.join(
                    self.get_code_dir(), 
                    config['features_to_use_path']
                )
        
        return config
    
    def create_experiment_dir(self, experiment_name):
        """Create and return experiment directory path."""
        experiment_dir = os.path.join(self.get_experiments_dir(), experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir