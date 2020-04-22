# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class Test:
    def __init__(self, false_neg_probability):
        self.false_neg_probability = false_neg_probability
    
    def run(self, samples):
        infected = False

        for sample in samples:
            if sample.infected:
                infected = True

        if np.random.rand() < self.false_neg_probability:
            infected = not infected
        
        return infected

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
        self.tested_positive = False

class TestProcedure:
    def __init__(self, batch, test_supplier, retries = 0):
        self.batch = batch
        self.test_supplier = test_supplier
        self.retries = retries
    
    def _count_infected_in_batch(self, batch):
        test = self.test_supplier.supply()
        if test.run(batch): # is infected
            if len(batch) == 1:
                batch[0].tested_positive = True
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


def run_simulation(batches, false_neg_probability, retries_on_negative):    
    test_supplier = TestSupplier(false_neg_probability)

    real_infected_count = 0
    tested_infected_count = 0
    pooled_infected_count = 0
    tested_false_negatives = 0
    pooled_false_negatives = 0
    
    for batch in batches:
        for sample in batch:
            test_result = Test(false_neg_probability).run([sample])
            if test_result:
                tested_infected_count = tested_infected_count + 1

            if sample.infected:
                real_infected_count = real_infected_count + 1
                if not test_result:
                    tested_false_negatives = tested_false_negatives + 1

            sample.tested_positive = False
          
        test_procedure = TestProcedure(batch, test_supplier, retries_on_negative)

        pooled_infected_count = pooled_infected_count + test_procedure.count_infected()

        for sample in batch:
            if sample.infected != sample.tested_positive:
                pooled_false_negatives = pooled_false_negatives + 1
    
    return {
        'pool_size': pool_size,
        'population': len(population),
        'infected': real_infected_count,
        'tested_pos': tested_infected_count,
        'pooled_pos': pooled_infected_count,
        'used_tests': test_supplier.tests_supplied,
        'tested_false_negatives' : tested_false_negatives,
        'pooled_false_negatives' : pooled_false_negatives
    }


false_negatives = [0.003]
#pool_sizes = [2 ** x for x in range(0, 7)]
pool_sizes = np.full(5, 32)

print(false_negatives)
print(pool_sizes)

population_size = 30000
infection_rate = 0.005

population = [Sample(infection_rate) for _ in range(0, population_size)]

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

    total_error = 0.0

    for pool_size in pool_sizes:
        batches = create_batches(population, pool_size)

        result_with_no_retries = run_simulation(batches, false_negative, 0)
        #print("No retries: %s error %.4f" % (result_with_no_retries, 100 * result_with_no_retries['pooled_false_negatives'] / result_with_no_retries['infected']))

        #total_error = total_error + (100 * result_with_no_retries['pooled_false_negatives'] / result_with_no_retries['infected'])
        
        result_with_one_retry = run_simulation(batches, false_negative, 1)
        print("One retry: %s error %.4f " % (result_with_one_retry, 100 * result_with_one_retry['pooled_false_negatives'] / result_with_one_retry['infected']))


        total_error = total_error + (100 * result_with_one_retry['pooled_false_negatives']  / result_with_one_retry['infected'])
        
        result_with_n_retries = run_simulation(batches, false_negative, pool_size - 1)
        #print("%d retries: %s error %.4f " % (pool_size - 1, result_with_no_retries, 100 * (result_with_n_retries['infected'] - result_with_n_retries['pooled_pos']) / result_with_n_retries['infected']))

        error_single_test.append(100 * (result_with_no_retries['infected'] - result_with_no_retries['tested_pos']) / result_with_no_retries['infected'])
        error_no_retries.append(100 * (result_with_no_retries['infected'] - result_with_no_retries['pooled_pos']) / result_with_no_retries['infected'])
        error_with_one_retry.append(100 * (result_with_one_retry['infected'] - result_with_one_retry['pooled_pos']) / result_with_one_retry['infected'])
        error_with_n_retries.append(100 * (result_with_n_retries['infected'] - result_with_n_retries['pooled_pos']) / result_with_n_retries['infected'])
        
       

    print ("Average error %.4f" % (total_error/len(pool_sizes))) 
    print ("False negative rate of tests: %.4f" % (100.0 * false_negative))
    plt.figure(figsize=(20, 10))
    plt.plot(pool_sizes, error_single_test, 'k-')
    plt.plot(pool_sizes, error_no_retries, 'ro')
    plt.plot(pool_sizes, error_with_one_retry, 'bo')
    plt.plot(pool_sizes, error_with_n_retries, 'g^')
    
    plt.ylim(ymin=0)
    #plt.show()