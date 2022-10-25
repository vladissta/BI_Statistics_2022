import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.weightstats import ztest


def fun_res(first_cell_type_expressions_path, second_cell_type_expressions_path, save_results_table):
    first_table = pd.read_csv(f"{first_cell_type_expressions_path}", index_col=0).select_dtypes(include=[np.number])
    second_table = pd.read_csv(f"{second_cell_type_expressions_path}", index_col=0).select_dtypes(include=[np.number])

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

        if l_2 > r_1 or l_1 > r_2:
            ci_test_results.append(True)
        else:
            ci_test_results.append(False)

        # z_test_results

        z, pval = ztest(first_table[i], second_table[i])
        z_test_results.append(pval < 0.05)
        z_test_p_values.append(pval)

        # mean_diff

        mean_diff.append(first_table[i].mean() - second_table[i].mean())

    results = {
        "genes": genes,
        "ci_test_results": ci_test_results,
        "z_test_results": z_test_results,
        "z_test_p_values": z_test_p_values,
        "mean_diff": mean_diff
    }

    results = pd.DataFrame(results)
    results.to_csv(f"{save_results_table}.csv")


fun_res(input('Path to the first table: '), input('Path to the second table: '), input('Name of results table: '))