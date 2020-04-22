# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import glob
import random

pd.set_option('display.max_columns', 5)

class Test:
    def __init__(self, false_neg_probability):
        self.false_neg_probability = false_neg_probability
    
    def run(self, samples):
        if np.random.rand() < self.false_neg_probability:
            return False
        else:
            for sample in samples:
                if sample.infected:
                    return True
            return False

class TestSupplier:
    
    tests_supplied = 0
    
    def __init__(self, false_neg_probability):
        self.false_neg_probability = false_neg_probability
    
    def supply(self):
        self.tests_supplied = self.tests_supplied + 1
        return Test(self.false_neg_probability)
    

class Sample:
    def __init__(self, p_infected):
        self.infected = (np.random.rand() < p_infected)

class TestProcedure:
    def __init__(self, batch, test_supplier, retries = 0):
        self.batch = batch
        self.test_supplier = test_supplier
        self.retries = retries
    
    def _count_infected_in_batch(self, batch):
        test = self.test_supplier.supply()
        if test.run(batch): # is infected
            if len(batch) == 1:
                return 1
            else:
                pivot = len(batch) // 2
                first_half = batch[:pivot]
                second_half = batch[pivot:]
                return self._count_infected_in_batch(first_half) + self._count_infected_in_batch(second_half)
        else:
            return 0
    
    def count_infected(self):
        infected_count = 0
        for retry in range(0, self.retries + 1):
            infected_count = self._count_infected_in_batch(self.batch)
            if infected_count > 0:
                break
        return infected_count

def create_batches(samples, size):
    batches = [samples[i:i + size] for i in range(0, len(samples), size)]
    return batches


def run_simulation(population, infection_rate, false_neg_probability, pool_size, retries_on_negative):    
    batches = create_batches(population, pool_size)
    test_supplier = TestSupplier(false_neg_probability)

    real_infected_count = 0
    tested_infected_count = 0
    pooled_infected_count = 0
    
    for batch in batches:
        for sample in batch:
            if sample.infected:
                real_infected_count = real_infected_count + 1
            if Test(false_neg_probability).run([sample]):
                tested_infected_count = tested_infected_count + 1
        
        test_procedure = TestProcedure(batch, test_supplier, retries_on_negative)
        pooled_infected_count = pooled_infected_count + test_procedure.count_infected()
    
    return {
        'false_neg_prob': false_neg_probability,
        'retries_on_negative': retries_on_negative,
        'pool_size': pool_size,
        'population': len(population),
        'infection_rate': infection_rate,
        'infected': real_infected_count,
        'tested_pos': tested_infected_count,
        'pooled_pos': pooled_infected_count,
        'used_tests': test_supplier.tests_supplied
    }

def run_simulations_round(population, infection_rate, test_runs, pool_sizes, false_negatives, retries_on_negative, out_prefix):
    for test_run in range(test_runs):
        shuffled_population = population.copy()
        random.shuffle(shuffled_population)
        results = []
        shuffle_results = []
        for pool_size in pool_sizes:
            for false_negative_rate in false_negatives:
                for retry_count in retries_on_negative:
                    results.append(run_simulation(population, infection_rate, false_negative_rate, pool_size, retry_count))
                    shuffle_results.append(run_simulation(shuffled_population, infection_rate, false_negative_rate, pool_size, retry_count))
                    
        with open('runs/out-noshuf-%s-%d.json' % (out_prefix, test_run), 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile)
        with open('runs/out-shuf-%s-%d.json' % (out_prefix, test_run), 'w', encoding='utf-8') as outfile:
            json.dump(shuffle_results, outfile)

infection_rates = [0.00004, 0.01, 0.0157]
#population_sizes = [30000]
population_sizes = [1300000]

def run_all_simulations():
    #false_negatives = np.arange(0.0, 0.4, 0.05)
    false_negatives = [0.0001 * (2 ** x) for x in range(0, 7)]
    #false_negatives = [0.0004]
    pool_sizes = [2 ** x for x in range(0, 8)]
    retries_on_negative = [0, 1, 2]
    test_runs = 100
    for infection_rate in infection_rates:
        for population_size in population_sizes:
            population = [Sample(infection_rate) for _ in range(population_size)]
            prefix = '%f-%d' % (infection_rate, population_size)
            run_simulations_round(population, infection_rate, test_runs, pool_sizes, false_negatives, retries_on_negative, prefix)
            

def load_runs_df(pattern='out-shuf-*.json'):
    df = pd.DataFrame()
    for file in glob.glob("runs/%s" % pattern):
        df = df.append(pd.read_json(file))
    return df


def save_as_csv():
    for infection_rate in infection_rates:
        for population_size in population_sizes:
            prefix = '%f-%d' % (infection_rate, population_size)
            load_runs_df('out-shuf-%s-*.json' % prefix).to_csv('out/results-shuf-%s.csv' % prefix)
            load_runs_df('out-noshuf-%s*.json' % prefix).to_csv('out/results-noshuf-%s.csv' % prefix)

def analyze_results():
    df = load_runs_df()   
    df = df[df.false_neg_prob.eq(0.0064)]
    
    df['pooled_err'] = (df['infected'] - df['pooled_pos']) / df['infected']
       
    def retry_group(df, retries):
        means = df[df.retries_on_negative.eq(retries)].groupby(['pool_size']).mean()
        stds = df[df.retries_on_negative.eq(retries)].groupby(['pool_size']).std()
        return (means, stds)
    
    plt.figure(figsize=(20, 10))

    means, stds = retry_group(df, 0)

    plt.errorbar(means.index, means['used_tests'], fmt='ro', yerr=stds['used_tests'])
    
    #plt.plot(means.index, means['infected'], 'b^')
    #plt.plot(means.index, means['pooled_pos'], 'r.')
    
    plt.ylim(ymin=0)
    
    
#run_all_simulations()

save_as_csv()

#analyze_results()

def _old():
    population = [Sample(infection_rate) for _ in range(0, population_size)]
    sorted_population = sorted(population, key = lambda s: s.infected)
    
    for false_negative in false_negatives:
        plt.figure()
        positive = []
    
        error_single_test = []
        error_no_retries = []
        tests_used_no_retries = []
        
        error_with_one_retry = []
        tests_used_one_retry = []
        
        error_with_n_retries = []
        tests_used_n_retries = []
        
        tests_used_regular = []
        tests_used_sorted = []
    
        for pool_size in pool_sizes:
            result_with_regular_population = run_simulation(population, false_negative, pool_size, 0)
            result_with_sorted_population = run_simulation(sorted_population, false_negative, pool_size, 0)
            
            tests_used_regular.append(result_with_regular_population['used_tests'])
            tests_used_sorted.append(result_with_sorted_population['used_tests'])
            
            print("Regular population: %s" % result_with_regular_population)
            print("Sorted population: %s" % result_with_sorted_population)
            
            #result_with_no_retries = run_simulation(population, false_negative, pool_size, 0)
            #print("No retries: %s" % result_with_no_retries)
            
            #result_with_one_retry = run_simulation(population, false_negative, pool_size, 1)
            #print("One retry: %s" % result_with_one_retry)
            
            #result_with_n_retries = run_simulation(population, false_negative, pool_size, pool_size - 1)
            #print("%d retries: %s" % (pool_size - 1, result_with_no_retries))
    
            #error_single_test.append(100 * (result_with_no_retries['infected'] - result_with_no_retries['tested_pos']) / result_with_no_retries['infected'])
            #error_no_retries.append(100 * (result_with_no_retries['infected'] - result_with_no_retries['pooled_pos']) / result_with_no_retries['infected'])
            #error_with_one_retry.append(100 * (result_with_one_retry['infected'] - result_with_one_retry['pooled_pos']) / result_with_one_retry['infected'])
            #error_with_n_retries.append(100 * (result_with_n_retries['infected'] - result_with_n_retries['pooled_pos']) / result_with_n_retries['infected'])
            
            
        print ("False negative rate of tests: %.5f" % false_negative)
        plt.figure(figsize=(20, 10))
        #plt.plot(pool_sizes, error_single_test, 'k-')
        #plt.plot(pool_sizes, error_no_retries, 'ro')
        #plt.plot(pool_sizes, error_with_one_retry, 'bo')
        #plt.plot(pool_sizes, error_with_n_retries, 'g^')
        
        plt.plot(pool_sizes, tests_used_regular, 'bo')
        plt.plot(pool_sizes, tests_used_sorted, 'r^')
        
        plt.ylim(ymin=0)
        plt.show()