"Must-Know information before trying to understand this code"
# rv => random variable
# of => observed frequency
# ef => expected frequency
# data = {rv1: of1, rv2: of2, ......................}
# table = {rv1: (of1, ef1), r2: (of2, ef2), ..................}



# Each "table_generator_for_"x"_dist function considers null hypothesis and alternate hypothesis as:
# H0 = The given data fits x distribution and H1 = The given data does not fit x distribution.
# then estimates sample parameters from the given data, and calculates the 
# expected frequencies of each random variable, (just like conditional probability): in this case,
# our condition is "H0 is True" 



from scipy.stats import binom
from scipy.stats import poisson


def table_generator_for_poisson_dist(data):

    random_variables = list(data.keys())
    frequencies = list(data.values())
    total_observations = sum(frequencies)
    

    total_sum = sum(k * f for k, f in data.items())
    if total_observations > 0:
        lam = total_sum / total_observations 
    else:
        lam = 0

    rv_ef_of_table = {
        k: (poisson.pmf(k, lam) * total_observations, data[k]) 
        for k in random_variables}
    
    return rv_ef_of_table


def table_generator_for_binomial_dist(data):
    
    random_variables = list(data.keys())
    frequencies = list(data.values())
    total_observations = sum(frequencies)
    n = max(random_variables)
    

    total_successes = sum(key * freq for key, freq in data.items())
    if total_observations > 0:
        p = total_successes / (n * total_observations) 
    else:
        p = 0

    rv_ef_of_table = {
        k: (binom.pmf(k, n, p) * total_observations, data[k]) 
        for k in random_variables}
    
    return rv_ef_of_table


def table_generator_for_geometric_dist(data):  
    
    random_variables = list(data.keys())
    frequencies = list(data.values())
    total_observations = sum(frequencies)  

    
    mean = sum(k * f for k, f in data.items()) / total_observations
    if mean > 0:
        p = 1 / mean
    else:
        p = 0 

    rv_ef_of_table = {}
    for k in random_variables:
        expected_frequency = total_observations * ((1 - p) ** (k - 1)) * p  
        rv_ef_of_table[k] = (expected_frequency, data[k])  
    
    return rv_ef_of_table


def table_generator_for_uniform_dist(data):

    random_variables = list(data.keys())
    frequencies = list(data.values())
    total_observations = sum(frequencies)

    no_of_random_variables = len(random_variables)  
    if no_of_random_variables > 0:
        p = 1 / no_of_random_variables    
    else:
        p = 0

    expected_frequency = p * total_observations
    rv_ef_of_table = {
        k: (expected_frequency, data[k]) 
        for k in random_variables}
    
    return rv_ef_of_table


# we need this "rv_ef_of_adjuster" function because we have to combine rows whose expected values are less than 5
# in order it to chi-squared distribution. That is called Chi-Squared Test Adjustment. It is a row merger 
# function who starts from the end row of our table and checks if expected frequency of the row is less than 5.
# if yes, it merges the row with it's preceeding row. If not, the pointer checks the row above and so on.


def rv_ef_of_adjuster(rv_ef_of_table):
    
    adjusted_rv_ef_of_table = {}
    current_combined_expected = 0
    current_combined_observed = 0
    combined_row = None

    for k in sorted(rv_ef_of_table.keys(), reverse=True):
        expected, observed = rv_ef_of_table[k]
        current_combined_expected += expected
        current_combined_observed += observed
        
        if combined_row is None:
            combined_row = str(k)
        else:
            combined_row = f"{k}-{combined_row}"
        
        if current_combined_expected >= 5:
            adjusted_rv_ef_of_table[combined_row] = (current_combined_expected, current_combined_observed)
            current_combined_expected = 0
            current_combined_observed = 0
            combined_row = None

    if current_combined_expected > 0:
        adjusted_rv_ef_of_table[combined_row] = (current_combined_expected, current_combined_observed)
    
    return adjusted_rv_ef_of_table


def calculate_chi_squared(adjusted_rv_ef_of_table):
    chi_squared_statistic = 0
    for expected, observed in adjusted_rv_ef_of_table.values():
        if expected > 0:
            chi_squared_statistic += (observed - expected) ** 2 / expected
    return chi_squared_statistic




def distfit(data):
    if len(data) < 1:
        print("Not Enough Data to work on")
        return
    
    distributions_and_their_test_statistics = {}
    distributions = {
        "Poisson Distribution": table_generator_for_poisson_dist,
        "Binomial Distribution": table_generator_for_binomial_dist,
        "Geometric Distribution": table_generator_for_geometric_dist,
        "Uniform Distribution": table_generator_for_uniform_dist,
    }

    print("Below are our Calculations\n")

    for name, generator in distributions.items():
        print(f"Analyzing data using {name}")
        adjusted_table = rv_ef_of_adjuster(generator(data))
        chi_squared_stat = calculate_chi_squared(adjusted_table)

        print()
        print(adjusted_table)
        print()
        print(chi_squared_stat)
        print("_" * 68)
        
        distributions_and_their_test_statistics[name] = chi_squared_stat

    print("\nDistribution with the lowest test statistic:")
    print(min(distributions_and_their_test_statistics, key=distributions_and_their_test_statistics.get))

data = {}
distfit(data)


