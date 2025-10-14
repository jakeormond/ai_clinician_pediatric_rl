"""
Analysis of health score, which we ultimately want to use as a reward signal. 
We need to confirm it is acting as intended. Values should correlate with mortality,
and we should see how it varies with different clinical features.
We also need to transform it into the reward signal and calculate discounted returns, 
making sure those look sensible.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu

def plot_health_score_distribution(survival_df, mortality_df, plot_dir):
    plt.figure(figsize=(10,6))
    plt.hist(survival_df['health_score_diff'], bins=50, alpha=0.5, label='Survived', color='g', density=True)
    plt.hist(mortality_df['health_score_diff'], bins=50, alpha=0.5, label='Died', color='r', density=True)
    plt.xlabel('Health Score')
    plt.ylabel('Density')
    plt.title('Health Score Distribution by Mortality')
    plt.legend()
    plt.grid(True)
    plt.show()
    # save figure
    plt.savefig(f"{plot_dir}/health_score_distribution.png")
    plt.close()

    # plot as violin plot
    plt.figure(figsize=(8,6))
    sns.violinplot(x='mortality', y='health_score_diff', data=pd.concat([survival_df, mortality_df]))
    plt.xlabel('Mortality (0=Survived, 1=Died)')
    plt.ylabel('Health Score')
    plt.title('Health Score Distribution by Mortality')
    plt.grid(True)
    plt.show()
    # save figure
    plt.savefig(f"{plot_dir}/health_score_violinplot.png")
    plt.close()


def stats_health_score(survival_df, mortality_df, stats_dir):
    # calculate the statistical difference between the two groups
    t_stat, t_p = ttest_ind(survival_df['health_score'], mortality_df['health_score'], equal_var=False)
    mw_stat, mw_p = mannwhitneyu(survival_df['health_score'], mortality_df['health_score'], alternative='two-sided')
    print(f"T-test: stat={t_stat:.3f}, p={t_p:.3e}")
    print(f"Mann-Whitney U test: stat={mw_stat:.3f}, p={mw_p:.3e}")
    # save to a json or txt file
    with open(f"{stats_dir}/health_score_stats.txt", "w") as f:
        f.write(f"T-test: stat={t_stat:.3f}, p={t_p:.3e}\n")
        f.write(f"Mann-Whitney U test: stat={mw_stat:.3f}, p={mw_p:.3e}\n")


def compare_time_stretched_trajectories(survival_df, mortality_df, plot_dir):
    # stretch each trajectory to 100 time points
    def stretch_trajectory(df, num_points=100):
        stretched = []
        for eid in df['encounter_id'].unique():
            traj = df[df['encounter_id'] == eid].sort_values('time')['health_score'].values
            if len(traj) < 2:
                continue
            stretched_traj = np.interp(np.linspace(0, len(traj)-1, num_points), np.arange(len(traj)), traj)
            stretched.append(stretched_traj)
        return np.array(stretched)
    
    survival_stretched = stretch_trajectory(survival_df)
    mortality_stretched = stretch_trajectory(mortality_df)

    # plot mean and std dev
    plt.figure(figsize=(12,6))
    plt.plot(np.mean(survival_stretched, axis=0), label='Survived', color='g')
    plt.fill_between(np.arange(survival_stretched.shape[1]), 
                     np.mean(survival_stretched, axis=0) - np.std(survival_stretched, axis=0),
                     np.mean(survival_stretched, axis=0) + np.std(survival_stretched, axis=0),
                     color='g', alpha=0.3)
    plt.plot(np.mean(mortality_stretched, axis=0), label='Died', color='r')
    plt.fill_between(np.arange(mortality_stretched.shape[1]), 
                     np.mean(mortality_stretched, axis=0) - np.std(mortality_stretched, axis=0),
                     np.mean(mortality_stretched, axis=0) + np.std(mortality_stretched, axis=0),
                     color='r', alpha=0.3)
    plt.xlabel('Stretched Time Points')
    plt.ylabel('Health Score')
    plt.title('Time-Stretched Health Score Trajectories')
    plt.legend()
    plt.grid(True)
    plt.show()
    # save figure
    plt.savefig(f"{plot_dir}/health_score_time_stretched.png")
    plt.close()



def analyse_health_score(dataset, config_manager):
    
    dataframe = dataset.dataframe[['encounter_id', 'time', 'mortality', 'health_score']].copy()
    
    # create health_score_diff as the difference in health score from previous time point
    dataframe['health_score_diff'] = dataframe['health_score'].diff()

    # shift the health_score_diff back one time point across the entire dataframe
    dataframe['health_score_diff'] = dataframe['health_score_diff'].shift(-1).fillna(0)


    encounter_ids = dataframe['encounter_id'].unique()
    mortality_ids = dataframe[dataframe['mortality'] == 1]['encounter_id'].unique()
    # survival_ids are encounter_ids not in mortality_ids
    survival_ids = np.array([eid for eid in encounter_ids if eid not in mortality_ids])

    survival_df = dataframe[dataframe['encounter_id'].isin(survival_ids)]
    mortality_df = dataframe[dataframe['encounter_id'].isin(mortality_ids)]
    mortality_df.loc[:, 'mortality'] = 1

    # plot the distributions
    plot_dir = config_manager.experiment_dir
    plot_health_score_distribution(survival_df, mortality_df, plot_dir)

    # visualize time stretched trajctories across all survivors and non-survivors
    compare_time_stretched_trajectories(survival_df, mortality_df, plot_dir)

    pass


def main():
    pass


if __name__ == "__main__":
    main()