import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.multitest import multipletests


def fun_res(first_cell_type_expressions_path, second_cell_type_expressions_path, save_results_table,
            adj_pval_method=None):
    first_table = pd.read_csv(f"{first_cell_type_expressions_path}", index_col=0).select_dtypes(include=[np.float64])
    second_table = pd.read_csv(f"{second_cell_type_expressions_path}", index_col=0).select_dtypes(include=[np.float64])

    genes = []
    ci_test_results = []
    z_test_results = []
    z_test_p_values = []
    mean_diff = []

    for i in first_table.columns:
        genes.append(i)

        # ci_test_results
        l_1, r_1 = st.t.interval(alpha=0.95,
                                 df=len(first_table[i]) - 1,
                                 loc=np.mean(first_table[i]),
                                 scale=st.sem(first_table[i]))

        l_2, r_2 = st.t.interval(alpha=0.95,
                                 df=len(second_table[i]) - 1,
                                 loc=np.mean(second_table[i]),
                                 scale=st.sem(second_table[i]))

        ci_test_results.append(l_2 > r_1 or l_1 > r_2)

        # z_test_results

        z, pval = ztest(first_table[i], second_table[i])
        z_test_results.append(pval < 0.05)
        z_test_p_values.append(pval)

        # mean_diff
        mean_diff.append(first_table[i].mean() - second_table[i].mean())

    # adjustment
    if adj_pval_method:
        z_test_p_values_adj = multipletests(z_test_p_values, alpha=0.05, method=adj_pval_method)[1]
        z_test_results = z_test_p_values_adj < 0.05

    results = {
        "genes": genes,
        "mean_diff_(table1-table2)": mean_diff,
        "ci_test_results": ci_test_results,
        "z_test_results": z_test_results,
        "z_test_p_values": z_test_p_values,
    }

    results = pd.DataFrame(results)

    if adj_pval_method:
        results['z_test_p_values_adj'] = z_test_p_values_adj

    results.to_csv(f"{save_results_table}.csv")


fun_res(input('Path to the first table: '), input('Path to the second table: '), input('Name of results table: '),
        adj_pval_method=input('Enter p_value adjustment method or press Enter if adjustment is not needed: '))
