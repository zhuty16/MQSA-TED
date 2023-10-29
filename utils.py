import numpy as np
import scipy.sparse as sp


def get_seq_matrix_sparse(train_dict, num_item, time_span):
    pairs = [[train_dict[u][n], train_dict[u][n+span]] for u in train_dict for span in range(1, time_span+1) for n in range(len(train_dict[u])-span)]
    # pairs += [[train_dict[u][n+span], train_dict[u][n]] for u in train_dict for span in range(1, time_span+1) for n in range(len(train_dict[u])-span)]
    pairs = np.array(pairs)
    seq_matrix_sparse = sp.csr_matrix(([1] * len(pairs), (pairs[:, 0], pairs[:, 1])), (num_item, num_item)).astype(np.float32)
    print("#seq pairs: %d" % len(pairs))
    return seq_matrix_sparse


def get_mask_matrix_sparse(train_dict, validate_dict, num_user, num_item):
    row_train = [u for u in train_dict for i in train_dict[u]]
    col_train = [i for u in train_dict for i in train_dict[u]]
    row_validate = [u for u in validate_dict]
    col_validate = [validate_dict[u] for u in validate_dict]
    mask_matrix_sparse_validate = sp.csr_matrix(([1] * len(row_train), (row_train, col_train)), (num_user, num_item)).astype(np.float32)
    mask_matrix_sparse_test = sp.csr_matrix(([1] * len(row_train + row_validate), (row_train + row_validate, col_train + col_validate)), (num_user, num_item)).astype(np.float32)
    return mask_matrix_sparse_validate, mask_matrix_sparse_test


def get_train_data(train_dict, num_item, max_len):
    train_data = list()
    for u in train_dict:
        input_seq = np.ones([max_len], dtype=np.int32) * num_item
        pred_seq = np.ones([max_len], dtype=np.int32) * num_item
        nxt = train_dict[u][-1]
        idx = max_len - 1
        for i in reversed(train_dict[u][:-1]):
            input_seq[idx] = i
            pred_seq[idx] = nxt
            nxt = i
            idx -= 1
            if idx == -1:
                break
        train_data.append([input_seq, pred_seq])
    return train_data


def get_train_batch(train_data, batch_size):
    train_batch = list()
    np.random.shuffle(train_data)
    i = 0
    while i < len(train_data):
        train_batch.append(np.array(train_data[i:i+batch_size]))
        i += batch_size
    return train_batch


def get_validate_data(train_dict, num_item, max_len):
    validate_data = list()
    for u in train_dict:
        input_seq = np.ones([max_len], dtype=np.int32) * num_item
        idx = max_len - 1
        for i in reversed(train_dict[u]):
            input_seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        validate_data.append(list(input_seq))
    validate_data = np.array(validate_data)
    return validate_data


def get_test_data(train_dict, validate_dict, num_item, max_len):
    test_data = list()
    for u in train_dict:
        input_seq = np.ones([max_len], dtype=np.int32) * num_item
        idx = max_len - 1
        input_seq[idx] = validate_dict[u]
        idx -= 1
        for i in reversed(train_dict[u]):
            input_seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        test_data.append(list(input_seq))
    test_data = np.array(test_data)
    return test_data


def get_feed_dict_train(model, batch_data_train, args, seq_matrix_sparse, num_item):
    feed_dict = dict()
    feed_dict[model.dropout_rate] = args.dropout_rate
    feed_dict[model.input_seq] = batch_data_train[:, 0]
    feed_dict[model.pred_seq] = batch_data_train[:, 1]
    item_head = np.array([i for i in np.reshape(batch_data_train[:, 0], -1) if i != num_item])
    feed_dict[model.item_head] = item_head
    feed_dict[model.item_tail] = seq_matrix_sparse[item_head, :].toarray()
    return feed_dict


def get_feed_dict_test(model, batch_data_test, batch_data_mask):
    feed_dict = dict()
    feed_dict[model.dropout_rate] = 0.0
    feed_dict[model.input_seq] = batch_data_test
    feed_dict[model.input_u_mask] = batch_data_mask
    return feed_dict


def get_top_K_index(pred_scores, K):
    ind = np.argpartition(pred_scores, -K)[:, -K:]
    arr_ind = pred_scores[np.arange(len(pred_scores))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_scores)), ::-1]
    batch_pred_list = ind[np.arange(len(pred_scores))[:, None], arr_ind_argsort]
    return batch_pred_list.tolist()
