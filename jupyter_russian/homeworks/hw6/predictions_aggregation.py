

import sys
import numpy as np
import pandas as pd


def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index=np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: predictions_aggregation.py file1.csv file2.csv")
        exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    print(file1, file2)
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    y_probas1 = df1['target'].values
    y_probas2 = df2['target'].values

    t = np.concatenate([y_probas1[:, None], y_probas2[:, None]], axis=1)
    y_probas = np.mean(t, axis=1)
    print(y_probas.shape)

    write_to_submission_file(y_probas,
                             "{}_{}.csv".format(file1.replace(".csv", ""),
                                                file2.replace(".csv", "")))
