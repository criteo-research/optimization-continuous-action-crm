# Libraries
import os
import os.path
import urllib.request
import pandas as pd 
import autograd.numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

dataset_dico = {
    'criteo': 'CriteoDataset',
    'noisycircles': 'Synthetic',
    'noisymoons': 'Synthetic',
    'anisotropic': 'Synthetic',
    'toy-gmm': 'Synthetic',
}

def get_dataset_by_name(name, random_seed):
    mod = __import__("utils.dataset", fromlist=[dataset_dico[name]])
    return getattr(mod, dataset_dico[name])(name=name, random_seed=random_seed)

class Dataset:
    """Parent class for Data
    
    """
    __metaclass__ = ABCMeta

    def __init__(self, random_seed=42, train_size=2/3, val_size=0.5):
        """Initializes the class
        
        Attributes:
            size (int): size of the dataset
            random_seed (int): random seed for randomized experiments
            rng (numpy.RandomState): random generator for randomization
            train_size (float): train/test ratio
            val_size (float): train/val ratio
            evaluation_offline (bool): perform evaluation offline only, for synthetic dataset this is False
            
        Note:
            Setup done in auxiliary private method
        """
        self.random_seed = random_seed
        self.rng = np.random.RandomState(42)
        self.train_size = train_size
        self.val_size = val_size
        self.evaluation_offline = True

    def get_data(self):
        """ Returns tuples of training and testing data
        """
        return (self.features_train, self.actions_train, self.reward_train, self.pi_0_train), \
               (self.features_valid, self.actions_valid, self.reward_valid, self.pi_0_valid), \
               (self.features_test, self.actions_test, self.reward_test, self.pi_0_test)

    @staticmethod
    def logging_policy(action, mu, sigma):
        """ Log-normal distribution PDF policy

        Args:
            action (np.array)
            mu (np.array): parameter of log normal pdf
            sigma (np.array): parameter of log normal pdf
        """
        return np.exp(-(np.log(action) - mu) ** 2 / (2 * sigma ** 2)) / (action * sigma * np.sqrt(2 * np.pi))

    def get_logging_policy_reward_baseline(self, mode='valid'):
        """
        Args
            mode (str): valid or test set
        """
        return self.baseline_reward_valid if mode =='valid' else self.baseline_reward_test

    def get_baseline_risk(self, mode='valid'):
        """
        Args
            mode (str): valid or test set
        """
        return - self.get_logging_policy_reward_baseline(mode)

    @abstractmethod
    def get_global_reward(self, features, potentials, estimator, parameter=None):
        """
        Args
            features (np.array): context for bandit feedbacks
            actions (np.array): actions taken by logging policy
            rewards (np.array): rewards associated to actions/context
            parameter (np.array):
        """
        pass

class CriteoDataset(Dataset):
    """ Criteo Off Policy Continuous Action Dataset
    
    """
    
    CRITEO_DATASET_URI = 'https://criteostorage.blob.core.windows.net/criteo-research-datasets/criteo-continuous-offline-dataset.csv.gz'
    CRITEO_DATASET_FILENAME = 'criteo-continuous-offline-dataset.csv.gz'
    
    file_name = 'criteo_dataset.npy'
    
    def __init__(self, name, path='data/', **kw):
        """Initializes the class
        
        Attributes:
            name (str): name of the dataset
            path (str): path for loading the dataset
            file_name (str): name of the file to load

        Note:
            Other attributed inherited from Dataset class
        """
        super(CriteoDataset, self).__init__(**kw)
        self.path = path
        self.file_path = os.path.join(self.path, self.file_name)
        self.name = name
        self._load_and_setup_data()

    def _download_criteo_open_dataset(self):
        local_csv = os.path.join(self.path, self.CRITEO_DATASET_FILENAME)
        if not os.path.exists(local_csv):
            print("downloading Criteo dataset (~2.3GB) to", local_csv, "...")
            urllib.request.urlretrieve(self.CRITEO_DATASET_URI,  local_csv)
        if not os.path.exists(self.file_path):
            print("converting Criteo dataset to numpy format...")        
            df = pd.read_csv(local_csv)
            array = df.values.astype(np.float64) 
            np.save(self.file_path, array)
            print("conversion done")
        
    def _load_and_setup_data(self):
        """ Load data from csv file
        """
        if not os.path.exists(self.file_path):
            self._download_criteo_open_dataset()
        
        data = np.load(self.file_path)
        
        features = data[:,:3]
        actions = data[:, 3]
        rewards = data[:, 4]
        pi_logging = data[:, 5]

        rng = np.random.RandomState(42)
        idx = rng.permutation(features.shape[0])
        features, actions, rewards, pi_logging = features[idx], actions[idx], rewards[idx], pi_logging[idx]

        size = int(features.shape[0] * 0.75)
        a_train, self.actions_test = actions[:size], actions[size:]
        f_train, self.features_test = features[:size, :], features[size:, :]
        r_train, self.reward_test = rewards[:size], rewards[size:]
        pi_0_train, self.pi_0_test = pi_logging[:size], pi_logging[size:]

        size = int(a_train.shape[0] * 2/3)
        self.actions_train, self.actions_valid = a_train[:size], a_train[size:]
        self.features_train, self.features_valid = f_train[:size, :], f_train[size:, :]
        self.reward_train, self.reward_valid = r_train[:size], r_train[size:]
        self.pi_0_train, self.pi_0_valid = pi_0_train[:size], pi_0_train[size:]

        self.baseline_reward_valid = np.mean(self.reward_valid)
        self.baseline_reward_test = np.mean(self.reward_test)


class Synthetic(Dataset):
    """Parent class for Data

    """
    __metaclass__ = ABCMeta

    def __init__(self, name, n_samples=30000, sigma=1, **kw):
        """Initializes the class

        Attributes:
            name (str): name of the dataset
            n_samples (int): number of samples
            start_mean (float): starting mean of the logging policy
            start_std (float): starting std of the logging policy
            start_sigma (float): starting parameter sigma of the logging policy
            start_mu (float): starting parameter mu of the logging policy
            mus (list): list of means of the potential group labels
            potentials_sigma (float): variance of the potential group labels

        Note:
            Setup done in auxiliary private method
        """
        super(Synthetic, self).__init__(**kw)
        self.name = name
        self.n_samples = n_samples
        self.start_mean = 2.
        self.start_std = sigma
        self.start_sigma = np.sqrt(np.log(self.start_std**2 / self.start_mean ** 2 + 1))
        self.start_mu = np.log(self.start_mean) - self.start_sigma ** 2 / 2
        self.mus = [3, 1, 0.1]
        self.potentials_sigma = 0.5
        self._setup()
        self.evaluation_offline = False

    def get_potentials_labels(self, mode):
        """
        Args
            mode (str): valid or test set
        """
        if mode=='train':
            return self.potentials_train, self.l_train
        elif mode=='test':
            return self.potentials_test, self.l_test
        else:
            return self.potentials_valid, self.l_valid

    def _get_potentials(self, y):
        """
        Args
            y (np.array): group labels

        """
        groups = [self.rng.normal(loc=mu, scale=self.potentials_sigma, size=self.n_samples) for mu in self.mus]
        potentials = np.ones_like(y, dtype=np.float64)
        for y_value, group in zip(np.unique(y), groups):
            potentials[y == y_value] = group[y == y_value]

        return np.abs(potentials)

    def get_X_y_by_name(self):
        if self.name == 'noisycircles':
            return datasets.make_circles(n_samples=self.n_samples, factor=.5,
                                         noise=.05, random_state=42)
        elif self.name == 'noisymoons':
            return datasets.make_moons(n_samples=self.n_samples, noise=.05, random_state=42)
        elif self.name == 'gmm':
            return datasets.make_blobs(n_samples=self.n_samples, random_state=42)
        elif self.name == 'anisotropic':
            X, y = datasets.make_blobs(n_samples=self.n_samples, centers=3, cluster_std=[[1/2, 1], [3/2, 1/2], [1, 3/2]], random_state=42)
            rng = np.random.RandomState(1)
            X = np.dot(X, rng.randn(2, 2))
            return X, y
        elif self.name == 'toy-gmm':
            X, y = datasets.make_blobs(centers=2, cluster_std=[[3, 1], [3, 2]],
                                n_samples=self.n_samples, random_state=42)
            return X, y
        else:
            return datasets.make_blobs(n_samples=self.n_samples, cluster_std=[1.0, 2.5, 0.5],
                                       random_state=42)

    def _setup(self):
        """ Setup the experiments and creates the data
        """
        # Actions
        features, y = self.get_X_y_by_name()
        potentials = self._get_potentials(y)
        actions = self.rng.lognormal(mean=self.start_mu, sigma=self.start_sigma, size=potentials.shape[0])
        rewards = self.get_rewards_from_actions(potentials, actions)
        pi_logging = Dataset.logging_policy(actions, self.start_mu, self.start_sigma)

        # Test train split
        self.actions_train, self.actions_test, self.features_train, self.features_test, self.reward_train, \
        self.reward_test, self.pi_0_train, self.pi_0_test, self.potentials_train, self.potentials_test, \
        self.l_train, self.l_test = train_test_split(actions, features, rewards, pi_logging, potentials, y,
                                                     train_size=self.train_size, random_state=42)

        self.actions_train, self.actions_valid, self.features_train, self.features_valid, self.reward_train, \
        self.reward_valid, self.pi_0_train, self.pi_0_valid, self.potentials_train, self.potentials_valid, \
        self.l_train, self.l_valid = train_test_split(self.actions_train, self.features_train, self.reward_train,
                                                      self.pi_0_train, self.potentials_train, self.l_train,
                                                     train_size=self.val_size, random_state=42)

        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        self.features_train = min_max_scaler.fit_transform(self.features_train)
        self.features_valid = min_max_scaler.transform(self.features_valid)
        self.features_test = min_max_scaler.transform(self.features_test)

        self.baseline_reward_valid = np.mean(self.reward_valid)
        self.baseline_reward_test = np.mean(self.reward_test)

    @staticmethod
    def get_rewards_from_actions(potentials, actions):
        return np.maximum(np.where(actions < potentials, actions/potentials, -0.5*actions+1+0.5*potentials), -0.1)

    def get_global_reward(self, features, potentials, estimator, n_samples=1):
        rewards = []
        for i in range(n_samples):
            actions_samples = estimator.get_samples(features, i)
            rewards += [self.get_rewards_from_actions(potentials, actions_samples)]
        rewards_array = np.stack(rewards, axis=0)
        var_pi = np.mean(np.var(rewards_array, axis=0))
        var_context = np.var(np.mean(rewards_array, axis=1))
        return np.mean(rewards_array), np.sqrt(var_pi), np.sqrt(var_context)

    def evaluation_online(self, metrics, mode, estimator, n_samples):
        """ Performs online evaluation

        Args:
            metrics (dic): metrics dictionnary to be filled
            mode (str): train, valid or test split
            estimator (estimator)
            n_samples (int): number of actions sampled for online evaluation

        Returns:
            metrics (dic): contains results information on the data split
        """
        features = self.features_valid if mode == 'valid' else self.features_test
        potentials = self.potentials_valid if mode == 'valid' else self.potentials_test
        mean_reward, std_pi, std_context = self.get_global_reward(features, potentials, estimator, n_samples)
        print("{} reward {:2f} policy std {:2f} context std {:2f}".format(mode, mean_reward, std_pi, std_context))
        metrics['{}_reward'.format(mode)] = mean_reward
        metrics['{}_policy_std'.format(mode)] = std_pi
        metrics['{}_test_context_std'.format(mode)] = std_context
        return metrics