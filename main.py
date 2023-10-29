import time
import argparse
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from MQSA_TED import MQSA_TED
from evaluate import evaluate, save_result
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="MQSA_TED")
    parser.add_argument("--dataset", type=str, default="beauty")
    # common hyperparameters
    parser.add_argument("--num_factor", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--random_seed", type=int, default=2023)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--N", type=int, default=1)
    # model-specific hyperparameters
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--num_block", type=int, default=1)
    parser.add_argument("--num_head", type=int, default=1)
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda_kd", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--time_span", type=int, default=1)
    args = parser.parse_args()
    for arg, arg_value in vars(args).items():
        print(arg, ":", arg_value)

    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    [train_dict, validate_dict, test_dict, num_user, num_item] = np.load("data/{dataset}/{dataset}.npy".format(dataset=args.dataset), allow_pickle=True)
    print("num_user: %d, num_item: %d" % (num_user, num_item))
    train_dict_len = [len(train_dict[u]) for u in train_dict]
    print("max len: %d, min len: %d, avg len: %.4f, med len: %.4f" % (np.max(train_dict_len), np.min(train_dict_len), np.mean(train_dict_len), np.median(train_dict_len)))
    train_dict = {u: train_dict[u][-args.max_len:] for u in train_dict}
    mask_matrix_sparse_validate, mask_matrix_sparse_test = get_mask_matrix_sparse(train_dict, validate_dict, num_user, num_item)
    seq_matrix_sparse = get_seq_matrix_sparse(train_dict, num_item, args.time_span)
    train_data = get_train_data(train_dict, num_item, args.max_len)
    validate_data = get_validate_data(train_dict, num_item, args.max_len)
    test_data = get_test_data(train_dict, validate_dict, num_item, args.max_len)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print("Model preparing...")
        model = MQSA_TED(num_user, num_item, args)
        sess.run(tf.global_variables_initializer())

        print("Model training...")
        result_validate = list()
        result_test = list()
        for epoch in range(1, args.num_epoch+1):
            t1 = time.time()
            train_loss = list()
            train_batch = get_train_batch(train_data, args.batch_size)
            for batch_data_train in train_batch:
                loss, loss_rec, loss_kd, _ = sess.run([model.loss, model.loss_rec, model.loss_kd, model.train_op], feed_dict=get_feed_dict_train(model, batch_data_train, args, seq_matrix_sparse, num_item))
                train_loss.append([loss, loss_rec, loss_kd])
            train_loss = np.mean(np.array(train_loss), 0)
            print("epoch: %d, %.2fs" % (epoch, time.time() - t1))
            print("training loss: %.4f, training loss_rec: %.4f, training loss_kd: %.4f" % tuple(train_loss))

            if epoch == 1 or epoch % args.N == 0:
                batch_size_test = args.batch_size
                rank_list = list()
                for start in range(0, num_user, batch_size_test):
                    test_logits = sess.run(model.test_logits, feed_dict=get_feed_dict_test(model, validate_data[start:start+batch_size_test], np.array(mask_matrix_sparse_validate[start:start+batch_size_test].todense())))
                    rank_list += get_top_K_index(test_logits, 20)
                rank_list = np.array(rank_list)
                result_validate.append([epoch] + evaluate(rank_list, validate_dict, 5) + evaluate(rank_list, validate_dict, 10) + evaluate(rank_list, validate_dict, 20))

                rank_list = list()
                for start in range(0, num_user, batch_size_test):
                    test_logits = sess.run(model.test_logits, feed_dict=get_feed_dict_test(model, test_data[start:start+batch_size_test], np.array(mask_matrix_sparse_test[start:start+batch_size_test].todense())))
                    rank_list += get_top_K_index(test_logits, 20)
                rank_list = np.array(rank_list)
                result_test.append([epoch] + evaluate(rank_list, test_dict, 5) + evaluate(rank_list, test_dict, 10) + evaluate(rank_list, test_dict, 20))
                #  We get the epoch of the best results on the validation set, and report the results of that epoch on the test set.

        save_result(args, result_validate, result_test)
