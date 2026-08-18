"""
Prepares data for supervised learning by creating a dataset class.
Will handle cross validation splits and data normalization.
Splits data into X and y (as opposed to the observations, actions,
and rewards in the RL setting).

Create a DataSet object that holds the full dataset, and then 
CVSplit objects that hold the encounter IDs for the splits
and methods to access the data in the DataSet object.
"""
import os
import argparse
import pandas as pd
import numpy as np
import time
from datetime import datetime
import wandb
import logging
import sys
from pathlib import Path

from src.data.data_processor_for_shards import ShardProcessor


from ...config_manager import ConfigManager  # Import the config manager
from ...supervised_learning.dataset import Dataset
# from ...supervised_learning.training_scikitlearn import outer_training_loop, \
#     aggregate_results_across_outer_folds, plot_saved_shap_results, plot_model_results
# from ...supervised_learning.models.dynamic_deephit import make_mixed_resolution_time_bins
# from ...supervised_learning.training import outer_training_loop, \
#     aggregate_results_across_outer_folds, plot_saved_shap_results, plot_model_results

    
from ...utils.logging_utilities import wandb_init  # Add this import

logger = logging.getLogger(__name__)

def setup_logger(verbose=False):
    """Configure root logging to print to stdout at INFO/DEBUG level.

    Uses force=True to override any prior configuration from imported libs.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description='Supervised learning')
    parser.add_argument('--hospital', 
                        default='choc', 
                        choices=['choc', 'mimic_iv', 'artificial'], 
                        help='Hospital dataset to use (default: choc)')
    
    parser.add_argument('--experiment_type', 
                        default=None, 
                        choices=['supervised_learning', 'reinforcement_learning'], 
                        help='Type of experiment to run (default: supervised_learning)')
    
    parser.add_argument('--model_type', 
                        default=None, 
                        choices=['xgbclassifier', 'xgbregressor', 'dynamic_deephit',
                                 'dynamic_deephit_transformer', 'dynamic_deephit_prediction_all',
                                'bc_transformer'], 
                        help='Model type to train or predict with') # both refers to generating predictions from both types
    
    parser.add_argument('--prediction_type', 
                        default=None, 
                        choices=['mortality_all', 'mortality_24', 'discharge_mortality', 
                                 'discharge_mortality_tte', 'action', 'behavioural_cloning'], 
                            help='type of prediction to make (default: mortality_all)') 

    parser.add_argument('--mode', type=str, default='single_fold',
                        choices=['save_dataframe', 'single_fold', 'aggregate', 'plot_shap', 'plot_model_results', 
                                 'single_split', 'development_single_split'],
                        help='Mode of training: single_fold or aggregate (default: single_fold)')
    
    parser.add_argument('--split_num', type=int, nargs='+', default=None,
                        help='Split number to train (not strictly required for single_split mode)')
    
    
    # mode-specific arguments    
    parser.add_argument('--fold', type=int, default=None,
                       help='Fold number to train (required for single_fold mode)')
    
    parser.add_argument('--experiment_folder', type=str, default=None,
                        help='Directory for experiment results (required for aggregate mode)')
    
    parser.add_argument('--final_timestep_only', type=str, default='False',
                        choices=['True', 'False'],
                        help='Whether to use only the final timestep for prediction (default: True)')
                        
    # Config file overrides (optional)
    parser.add_argument('--action_mode', 
                    type=str, 
                    default = "continuous_only",   
                    choices=['continuous_only', 'binary_only', 'mixed'],               
                    help='Type of action space to use - continuous_only, binary_only, or mixed')


    parser.add_argument('--experiment_variant', type=str, default='standard',
                       choices=['standard', 'quick_test', 'full_search', 'prediction', 'feature_evaluation'],
                       help='Experiment variant determines config defaults')

    parser.add_argument('--search_variant', type=str, default=None,
                        choices=['grid_search', 'random_search', 'fixed_hyperparameters'],
                        help='Search variant for hyperparameter tuning (default: None)')
    
    parser.add_argument('--n_search_iterations', type=int, default=0,
                        help='Number of search iterations for random search (default: 0)')
    

    parser.add_argument('--cohort', type=str, default=None,
                        choices=['all', 'sepsis_strict', 'sepsis_broad', 'sepsis',
                            'ards_strict', 'ards_broad', 'aki_strict', 'aki_broad'],
                        help='Cohort to use for this experiment (overrides config file)')


    parser.add_argument('--unit', type=str, default=None,
                        choices=['PICU', 'NICU', 'ICU'],
                        help='Unit to use for this experiment (overrides config file)')
    

    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory of the model to load for prediction or feature evaluation (required for prediction and feature_evaluation experiment variants)')


    parser.add_argument('--start_time', type=str, default = None,                  
                    help='Start time for this experiment run')
    
    parser.add_argument('--testing', action='store_true',                 
                    help='if true, we pare down the data for quick testing')

    parser.add_argument('--verbose', action='store_true',                 
                    help='if true, print details of all the features and normalization steps')

    parser.add_argument('--calculate_shap', type=str, default='false',
                        choices=['true', 'false'],
                        help='Whether to calculate SHAP values (default: false)')
    
    parser.add_argument('--shap_max_samples', type=int, default=5000,
                        help='Maximum number of samples to use for SHAP analysis (default: None - use all)')
    
    parser.add_argument('--shap_background_samples', type=int, default=100,
                        help='Number of background samples for SHAP analysis (default: 100)')
    
    parser.add_argument('--n_epochs', type=int, default = 10,                  
                    help='Number of epochs to train for - should be 1000, but for testing we set to 10')  
    
    # seeds is a list if not None, but argparse doesn't support list of ints, so we use nargs='+' and convert to list of ints
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
        help='Random seeds to use for training (default: None, which means use the default seed in the config file)')
        


    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_id = (os.environ.get("SLURM_JOB_ID")
                or os.environ.get("PBS_JOBID")
                or os.getpid())
    args.start_time = f"{ts}_job{job_id}"

    if len(args.split_num) == 1: args.split_num = args.split_num[0]
    
    return args


def validate_arguments(args):
    if args.mode == "single_fold" and args.fold is None:
        raise ValueError("--fold is required when mode is 'single_fold'")
    if args.mode == "aggregate" and args.experiment_folder is None:
        raise ValueError("--experiment_folder is required when mode is 'aggregate'")


def log_fold_results_to_wandb(results):
    """Log fold results to W&B."""
    if wandb.run is None:  # Check if W&B is active
        return
    
    fold = results['fold']
    
    # Use global wandb state
    wandb.log({
        f"fold_{fold}/best_cv_score": results['best_cv_score'],
        **{f"fold_{fold}/{k}": v for k, v in results['test_metrics'].items()}
    }, step=fold)


def main():
    args = parse_arguments()
    # Configure logging before other modules potentially emit logs
    setup_logger(verbose=args.verbose)
    logger.info("Logging configured (level=%s)", logging.getLevelName(logging.getLogger().getEffectiveLevel()))
    validate_arguments(args)


    #################### PREDICTIONS ON TRAINED MODELS ####################
    if args.experiment_variant == 'prediction' and args.split_num is not None:
        """
        this is for generating predictions on the MIMIC IV dataset with a single train/val/test split, for a specific split number.
        """
        ##################### DYNAMIC_DEEPHIT_TRANSFORMER PREDICTIONS #####################
        if args.model_type == "dynamic_deephit_transformer":
            from ...supervised_learning.predict import get_predictions, get_all_predictions, bootstrap_evaluation, plot_random_predictions, plot_health_score_statistics

            split_nums = args.split_num if isinstance(args.split_num, list) else [args.split_num]
            
            dynamic_deephit_datasets = []
            for split_num in split_nums:
                args.split_num = split_num

                config_manager = ConfigManager(args)
                # set training config
                config_manager.set_training_config()

                # manually set tuning to false (it may or may not be set false already)
                config_manager.training_config['tuning'] = False
                
                # set max encounter length as 2 months
                max_encounter_length = 24 * 60  # hours

                if config_manager.hospital == "mimic_iv":
                    time_bins = [0, 13, 28, 47, 75, 116, 185, 322, 1435]
                elif config_manager.hospital == "choc":

                    # calculated in get_time_bins.ipynb based on the distribution of trajectory lengths in the CHOC dataset, with a max trajectory length of 60 days (1440 hours)
                    ####### if using all stays ###########
                    # time_bins = [0, 45, 94, 154, 230, 329, 472, 696, 1441]

                    # if using only PICU sepsis stays - see get_time_bins.ipynb
                    time_bins = [0, 50, 103, 164, 237, 328, 460, 669, 1398]

                else:            
                    time_bins = make_mixed_resolution_time_bins(max_time_hours=max_encounter_length)
                config_manager.max_encounter_length = max_encounter_length
                config_manager.time_bins = time_bins  

                print(f"Running on {config_manager.environment} environment")    

                logger.info(f"Processing split {split_num}")
                config_manager.split_num = split_num

                model_type = config_manager.model_type
                experiment_dir = config_manager.experiment_dir


                predictions = get_all_predictions(
                    config_manager, 
                    model_type, 
                    experiment_dir
                ) # generates the predictions, need to modify to get entropy and health score

                # dynamic_deephit_datasets.append(dynamic_deephit_dataset)

                # bootstrap_evaluation(
                #     dynamic_deephit_dataset,
                #     config_manager=config_manager,
                #     n_bootstrap_samples=100
                # ) # generates the bootstrap evaluation results for the predictions

                
                plot_health_score_statistics(config_manager)
                plot_random_predictions(config_manager)
        
            # after generating predictions for all specified splits, we can aggregate the results
            # save_dir = Path(config_manager.experiment_dir).parents[2]

            # combined_dynamic_deephit_dataset = ShardProcessor.concatenate_dynamic_deephit_datasets(dynamic_deephit_datasets, save_dir = save_dir)
                    
            return
        
        
        ####################### BC_TRANSFORMER PREDICTIONS #######################
        elif args.model_type == "bc_transformer":
            from ...supervised_learning.predict_behavioural_cloning_transformer import plot_random_predictions, get_all_predictions

            split_nums = args.split_num if isinstance(args.split_num, list) else [args.split_num]
                
            dynamic_deephit_datasets = []
            for split_num in split_nums:
                args.split_num = split_num

                config_manager = ConfigManager(args)
                # set training config
                config_manager.set_training_config()

                # manually set tuning to false (it may or may not be set false already)
                config_manager.training_config['tuning'] = False
                
                # set max encounter length as 2 months
                max_encounter_length = 24 * 60  # hours
                config_manager.max_encounter_length = max_encounter_length

                print(f"Running on {config_manager.environment} environment")    

                logger.info(f"Processing split {split_num}")
                config_manager.split_num = split_num

                model_type = config_manager.model_type
                experiment_dir = config_manager.experiment_dir

                actions_in_state = config_manager.training_config['actions_in_state']
                if actions_in_state:
                    # insert subdirectory "with_actions" before "final_model" in the experiment_dir path
                    experiment_dir = Path(
                        str(experiment_dir).replace("/final_model", "/with_actions/final_model/")
                    )
                else:
                    experiment_dir = Path(
                        str(experiment_dir).replace("/final_model", "/without_actions/final_model/")
                    )

                # truncate experiment_dir after final_model
                experiment_dir = str(experiment_dir).split("final_model")[0] + "final_model/"

                # find the folder inside experiment_dir and add it to experiment_dir
                subfolders = [f.path for f in os.scandir(experiment_dir) if f.is_dir()]
                if len(subfolders) == 0:
                    raise ValueError(f"No subfolders found in {experiment_dir}")
                elif len(subfolders) > 1:
                    raise ValueError(f"Multiple subfolders found in {experiment_dir}: {subfolders}")
                else:
                    experiment_dir = subfolders[0]

                config_manager.experiment_dir = experiment_dir

                print(f"Experiment directory for predictions: {config_manager.experiment_dir}")
                predictions = get_all_predictions(
                    config_manager, 
                    model_type, 
                    experiment_dir
                )

                plot_random_predictions(config_manager)

                return



    # Initialize config manager, it holds paths and environment info
    config_manager = ConfigManager(args)
    if config_manager.model_type == "dynamic_deephit" or config_manager.model_type == "dynamic_deephit_transformer" or \
        config_manager.model_type == "dynamic_deephit_prediction_all" or config_manager.model_type == "bc_transformer": 
        from ...supervised_learning.training import single_split_training

    if config_manager.model_type == "dynamic_deephit" or config_manager.model_type == "dynamic_deephit_transformer" or \
        config_manager.model_type == "dynamic_deephit_prediction_all":         
        from ...supervised_learning.models.dynamic_deephit import make_mixed_resolution_time_bins
        from ...supervised_learning.predict import predict, plot_evaluation, predict_all, plot_random_predictions

    else:
        from ...supervised_learning.training_scikitlearn import outer_training_loop, \
            aggregate_results_across_outer_folds, plot_saved_shap_results, plot_model_results

    if args.mode == 'save_dataframe': # this is for post-hoc analysis
        dataset = Dataset(config_manager.data_config, drop_actions=False, transform_target=False)
        dataframe_path = dataset.save_dataframe(config_manager)
     
    
    elif args.mode == 'single_split':
        if (config_manager.model_type == "dynamic_deephit" or config_manager.model_type == "dynamic_deephit_transformer" or \
            config_manager.model_type == "dynamic_deephit_prediction_all" or config_manager.model_type == "bc_transformer"): 
            # set training config
            config_manager.set_training_config()

        if (config_manager.model_type == "dynamic_deephit" or config_manager.model_type == "dynamic_deephit_transformer" or \
            config_manager.model_type == "dynamic_deephit_prediction_all"): 
                                
            # set max encounter length as 2 months
            max_encounter_length = 24 * 60  # hours - CONFIRM THIS IS CODED IN THE DATA PREP
            
            if config_manager.hospital == "mimic_iv":
                time_bins = [0, 13, 28, 47, 75, 116, 185, 322, 1435]
            elif config_manager.hospital == "choc":
                # calculated in get_time_bins.ipynb based on the distribution of trajectory lengths in the CHOC dataset, with a max trajectory length of 60 days (1440 hours)
                # using all data
                # time_bins = [0, 45, 94, 154, 230, 329, 472, 696, 1441]
                # using picu sepsis under 19 stays
                time_bins = [0, 50, 103, 164, 237, 328, 460, 669, 1398]
            else:            
                # time_bins = make_mixed_resolution_time_bins(max_time_hours=max_encounter_length)
                raise ValueError(f"Time bins for hospital {config_manager.hospital} not defined")
            config_manager.max_encounter_length = max_encounter_length
            config_manager.time_bins = time_bins  

            print(f"Running on {config_manager.environment} environment")         
        
        
        if args.experiment_variant != 'prediction' and args.experiment_variant != 'feature_evaluation':      
            """
            this is for training on the MIMIC IV dataset with a single train/val/test split.
            Because the data is in hdf5 shards, we can't do much processing here, 
            instead the data processopr for shards will handle it
            """      
            print(f"Experiment directory: {config_manager.experiment_dir}")
        
            results = single_split_training(config_manager)

       
        elif args.experiment_variant == 'prediction' and args.cohort == 'all':
            """
            this is for generating predictions on the MIMIC IV dataset with a single train/val/test split.
            """
            cohorts = ['aki_broad', 'aki_strict', 'ards_broad', 'ards_strict',
                       'sepsis_broad', 'sepsis_strict']
            
            
            models = ['transformer']
            # cohort comparisons
            predict_all(
                config_manager, 
                n_bootstrap_samples=100,
                cohorts=cohorts,
                models=models,
                outer_group='models',
                inner_group='cohorts',
                base_filename='cohort_comparison'
            )
            
            
            # models = ['GRU', 'transformer']
            
            # predict_all(
            #     config_manager, 
            #     n_bootstrap_samples=100,
            #     cohorts=['all'],
            #     models=models,
            #     outer_group='cohorts',
            #     inner_group='models',
            #     base_filename='all_stays_model_comparison'
            # )

            # models = ['GRU', 'transformer']
            # predict_all(
            #     config_manager, 
            #     n_bootstrap_samples=100,
            #     cohorts=['all'],
            #     models=models,
            #     outer_group='cohorts',
            #     inner_group='models',
            #     base_filename='all_stays_model_comparison',
            #     time_since_admission='0-24'
            # )

            # models = ['GRU', 'transformer']
            # predict_all(
            #     config_manager, 
            #     n_bootstrap_samples=100,
            #     cohorts=['all'],
            #     models=models,
            #     outer_group='cohorts',
            #     inner_group='models',
            #     base_filename='all_stays_model_comparison',
            #     time_since_admission='24-72'
            # )

            # models = ['GRU', 'transformer']
            # predict_all(
            #     config_manager, 
            #     n_bootstrap_samples=100,
            #     cohorts=['all'],
            #     models=models,
            #     outer_group='cohorts',
            #     inner_group='models',
            #     base_filename='all_stays_model_comparison',
            #     time_since_admission='72+'
            # )
                      
            # plot_evaluation(
            #     config_manager,
            #     cohorts=['all'],
            #     models=['GRU', 'transformer'],
            #     outer_group='cohorts',
            #     inner_group='models',
            #     base_filename='all_stays_model_comparison') 
            
            # models = ['transformer']
            # predict_all(
            #     config_manager, 
            #     n_bootstrap_samples=100,
            #     cohorts=cohorts,
            #     models=models,
            #     outer_group='models',
            #     inner_group='cohorts',
            #     base_filename='cohort_model_comparison'
            # )

            models = ['transformer']
            predict_all(
                config_manager, 
                n_bootstrap_samples=100,
                cohorts=cohorts,
                models=models,
                outer_group='models',
                inner_group='cohorts',
                base_filename='cohort_model_comparison',
                time_since_admission='0-24'
            )

            predict_all(
                config_manager, 
                n_bootstrap_samples=100,
                cohorts=cohorts,
                models=models,
                outer_group='models',
                inner_group='cohorts',
                base_filename='cohort_model_comparison',
                time_since_admission='24-72'
            )

            predict_all(
                config_manager, 
                n_bootstrap_samples=100,
                cohorts=cohorts,
                models=models,
                outer_group='models',
                inner_group='cohorts',
                base_filename='cohort_model_comparison',
                time_since_admission='72+'
            )

            plot_evaluation(
                config_manager,
                cohorts=cohorts,
                models=['transformer'],
                outer_group='models',
                inner_group='cohorts',
                base_filename='cohort_model_comparison')

            plot_random_predictions(config_manager)   
                   
        
        elif args.experiment_variant == 'feature_evaluation':
            """
            this is for generating feature attributions on the MIMIC IV dataset with a single train/val/test split.
            """
            from ...supervised_learning.dynamic_deephit_feature_evaluation import get_feature_attributions

            sample_size = 100
            get_feature_attributions(
                config_manager,
                n_batches=sample_size
            )
            
    
    elif args.mode == 'development_single_split':
      
        # load data
        dataset = Dataset(config_manager.data_config)
        print(f"Dataset loaded with {len(dataset.dataframe)} rows and {len(dataset.dataframe.columns)} columns")

        # set training config
        config_manager.set_training_config()

        if config_manager.testing:
            dataset = dataset.subsample_dataset(0.1)
            # print(f"Subsampled dataset to {len(dataset.dataframe)} rows for testing")   
            # dataset = dataset.truncate_trajectories(max_length = 72, window_length=config_manager.training_config['window_length']) # hours
            
            # truncate to 7 days, and try and predict the day of event
            dataset = dataset.truncate_trajectories(max_length = 168, window_length=config_manager.training_config['window_length']) # hours
        
        if config_manager.model_type == "dynamic_deephit":
            # get lengths of trajectories in dataset.dataframe (by encounter_id)
            encounter_lengths = dataset.dataframe.groupby("encounter_id").size()
            max_encounter_length = encounter_lengths.max()
            print(f"Max encounter length: {max_encounter_length}")

            time_bins = make_mixed_resolution_time_bins(max_time_hours=max_encounter_length)
            config_manager.time_bins = time_bins

        print(f"Running on {config_manager.environment} environment")
        print(f"Experiment directory: {config_manager.experiment_dir}")

        results = development_training_loop(dataset, config_manager)

        
    elif args.mode == "single_fold":        
        # set base experiment name and make directory, if it doesn't exist (it will 
        # already if another folder was run)
        # config_manager.set_experiment_name()
        
        # # update config with n_folds and current fold 
        # config_manager.set_cv_config() # sets the number of outer and inner folds
        # config_manager.add_fold(args.fold)
        
        # create the name and directory for this fold
        config_manager.set_fold_name()

        # # update config with the search space
        # config_manager.set_search_grid()      

        # load data
        dataset = Dataset(config_manager.data_config)
        print(f"Dataset loaded with {len(dataset.dataframe)} rows and {len(dataset.dataframe.columns)} columns")

        # Initialize W&B
        # wandb_init(config_manager.wandb_config)

        # load training config
        config_manager.set_training_config() # based on model type 

        if config_manager.testing:
            dataset = dataset.subsample_dataset(0.1)
            print(f"Subsampled dataset to {len(dataset.dataframe)} rows for testing")   
            # dataset = dataset.truncate_trajectories(max_length = 72, window_length=config_manager.training_config['window_length']) # hours
            
            # truncate to 7 days, and try and predict the day of event
            dataset = dataset.truncate_trajectories(max_length = 168, window_length=config_manager.training_config['window_length']) # hours
            print(f"Truncated trajectories to max length of 168 hours for testing")
            
        # if model type is deep hit, we need to make the time bins and add to the configuration
        if config_manager.model_type == "dynamic_deephit":
            # get lengths of trajectories in dataset.dataframe (by encounter_id)
            encounter_lengths = dataset.dataframe.groupby("encounter_id").size()
            max_encounter_length = encounter_lengths.max()
            print(f"Max encounter length: {max_encounter_length}")

            time_bins = make_mixed_resolution_time_bins(max_time_hours=max_encounter_length)
            config_manager.time_bins = time_bins



        print(f"Running on {config_manager.environment} environment")
        print(f"Experiment directory: {config_manager.experiment_dir}")

        results = outer_training_loop(dataset, config_manager)

        # LOG TO W&B in main script
        # log_fold_results_to_wandb(results)

        print(f"Fold {args.fold} completed successfully")
        # print(f"Test metrics: {results['test_metrics']}")
        
        # if wandb.run is not None:
        #     wandb.finish()

    elif args.mode == "aggregate":
        # aggregate results from all folds
        results = aggregate_results_across_outer_folds(config_manager)

    elif args.mode == "plot_model_results":
        # load config and all_predictions_df
        all_preductions_path = os.path.join(config_manager.get_experiments_dir,
            config_manager.experiment_folder, "all_predictions.parquet")
        all_predictions_df = pd.read_parquet(all_preductions_path, engine='pyarrow')

        # config_manager saved as pkl
        config_path = os.path.join(config_manager.get_experiments_dir,
            config_manager.experiment_folder, "config.pkl")
        config_manager = pd.read_pickle(config_path)

        plot_model_results(all_predictions_df, config_manager)

    
    elif args.mode == "plot_shap":
        # shap_result_path = config_manager.get_experiments_dir + 
        # config_manager.experiment_folder + "/shap_analysis/"
        shap_results_path = os.path.join(config_manager.get_experiments_dir,
            config_manager.experiment_folder, "shap_analysis.pkl")
        plot_saved_shap_results(shap_results_path)
    
    
    elif args.mode == "combine_aggregates":

        results = aggregate_results_across_outer_folds(config_manager)

    

if __name__ == "__main__":
    main()