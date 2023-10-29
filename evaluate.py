import math
import csv
import numpy as np


def evaluate(rank_list, test_dict, K):
    rank_list = rank_list[:, :K]
    hit_ratio = 0
    ndcg = 0
    for user in range(len(rank_list)):
        rec_list = list(rank_list[user])
        ground_truth = test_dict[user]
        if ground_truth in rec_list:
            hit_ratio += 1
            ndcg += math.log(2) / math.log(rec_list.index(ground_truth) + 2)
    hit_ratio_avg = hit_ratio / len(rank_list)
    ndcg_avg = ndcg / len(rank_list)
    print("HR@{K}: %.4f, NDCG@{K}: %.4f".format(K=K) % (hit_ratio_avg, ndcg_avg))
    return [hit_ratio_avg, ndcg_avg]


def save_result(args, result_valid, result_test):
    ndcg_20 = list(np.array(result_valid)[:, 6])
    ndcg_20_max = max(ndcg_20)
    result_report = result_test[ndcg_20.index(ndcg_20_max)]
    #  We get the epoch of the best results on the validation set, and report the results of that epoch on the test set.

    args_dict = vars(args)
    filename = ""
    for arg in args_dict:
        filename += str(args_dict[arg]) + "_"
    with open(filename + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "HR@5", "NDCG@5", "HR@10", "NDCG@10", "HR@20", "NDCG@20"])
        for line in result_test:
            writer.writerow(line)
        writer.writerow(result_report)
        for arg in args_dict:
            writer.writerow(["", arg, args_dict[arg]] + [""] * (len(line) - 3))
