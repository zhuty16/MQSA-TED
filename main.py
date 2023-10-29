import time
import argparse
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from evaluate import evaluate, save_result
from utils import *


class MQSA_TED(object):
    def __init__(self, num_user, num_item, args):
        self.num_user = num_user
        self.num_item = num_item

        self.num_factor = args.num_factor
        self.l2_reg = args.l2_reg
        self.lr = args.lr
        self.max_len = args.max_len
        self.L = args.L
        self.num_block = args.num_block
        self.num_head = args.num_head

        self.dropout_rate = tf.placeholder(tf.float32)
        self.input_seq = tf.placeholder(tf.int32, [None, self.max_len], name="input_seq")
        self.pred_seq = tf.placeholder(tf.int32, [None, self.max_len], name="pred_seq")

        self.item_head = tf.placeholder(tf.int32, [None], name="item_head")
        self.item_tail = tf.placeholder(tf.float32, [None, self.num_item], name="item_tail")

        self.mask = tf.expand_dims(tf.cast(tf.not_equal(self.input_seq, self.num_item), tf.float32), -1) # [batch_size, max_len, 1]

        with tf.name_scope("item_embedding"):
            item_embedding_ = tf.Variable(tf.random_normal([self.num_item, self.num_factor], stddev=0.01), name="item_embedding")
            item_embedding = tf.concat([item_embedding_, tf.zeros([1, self.num_factor], dtype=tf.float32)], 0)

        with tf.name_scope("positional_embedding"):
            position = tf.tile(tf.expand_dims(tf.range(self.max_len), 0), [tf.shape(self.input_seq)[0], 1])
            position_embedding = tf.Variable(tf.random_normal([self.max_len, self.num_factor], stddev=0.01), name="position_embedding")
            p_emb = tf.nn.embedding_lookup(position_embedding, position)
            seq_emb = tf.nn.dropout(tf.nn.embedding_lookup(item_embedding, self.input_seq) * (self.num_factor ** 0.5) + p_emb, keep_prob=1-self.dropout_rate) * self.mask

        with tf.name_scope("block_short_query"):
            seq_emb_short = tf.identity(seq_emb)
            for _ in range(self.num_block):
                # Self-attention
                # Linear projections
                seq = seq_emb_short
                seq_norm = self.layer_normalize(seq)
                Q = tf.layers.dense(seq_norm, self.num_factor, activation=None)
                K = tf.layers.dense(seq, self.num_factor, activation=None)
                V = tf.layers.dense(seq, self.num_factor, activation=None)

                # Split and concat
                Q_ = tf.concat(tf.split(Q, self.num_head, axis=2), axis=0)
                K_ = tf.concat(tf.split(K, self.num_head, axis=2), axis=0)
                V_ = tf.concat(tf.split(V, self.num_head, axis=2), axis=0)

                # Multiplication
                outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

                # Scale
                outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

                # Key Masking
                key_masks = tf.sign(tf.reduce_sum(tf.abs(seq), axis=-1))
                key_masks = tf.tile(key_masks, [self.num_head, 1])
                key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(seq_norm)[1], 1])

                paddings = tf.ones_like(outputs)*(-2**32+1)
                outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

                # Causality (Future blinding)
                diag_vals = tf.ones_like(outputs[0, :, :])
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
                #tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # for earlier tf versions
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

                paddings = tf.ones_like(masks)*(-2**32+1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

                # Activation
                outputs = tf.nn.softmax(outputs)

                # Query Masking
                query_masks = tf.sign(tf.reduce_sum(tf.abs(seq_norm), axis=-1))
                query_masks = tf.tile(query_masks, [self.num_head, 1])
                query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(seq)[1]])
                outputs *= query_masks

                # Dropouts
                outputs = tf.nn.dropout(outputs, keep_prob=1-self.dropout_rate)

                # Weighted sum
                outputs = tf.matmul(outputs, V_)

                # Restore shape
                outputs = tf.concat(tf.split(outputs, self.num_head, axis=0), axis=2)

                # Residual connection
                outputs += seq_norm

                # Layer normalization
                outputs = self.layer_normalize(outputs)

                # Feed forward
                # Layer 1
                outputs_ = tf.layers.dense(outputs, self.num_factor, activation=tf.nn.relu, use_bias=True)
                outputs_ = tf.nn.dropout(outputs_, keep_prob=1-self.dropout_rate)

                # Layer 2
                outputs_ = tf.layers.dense(outputs_, self.num_factor, activation=None, use_bias=True)
                outputs_ = tf.nn.dropout(outputs_, keep_prob=1-self.dropout_rate)

                # Residual connection
                outputs += outputs_

                seq_emb_short = outputs * self.mask

            seq_emb_short = self.layer_normalize(seq_emb_short)

        with tf.name_scope("block_long_query"):
            seq_emb_long = tf.identity(seq_emb)
            for _ in range(self.num_block):
                # Self-attention
                # Linear projections
                seq = seq_emb_long
                seq_avg = tf.identity(seq)
                mask_avg = tf.identity(self.mask)
                for l in range(1, self.L):
                    seq_avg += tf.concat([tf.zeros([tf.shape(self.input_seq)[0], l, self.num_factor], dtype=tf.float32), seq[:, :-l, :]], 1)
                    mask_avg += tf.concat([tf.zeros([tf.shape(self.input_seq)[0], l, 1], dtype=tf.float32), self.mask[:, :-l, :]], 1)
                seq_avg = seq_avg / (mask_avg + 1e-24)
                seq_avg_norm = self.layer_normalize(seq_avg)

                Q = tf.layers.dense(seq_avg_norm, self.num_factor, activation=None)
                K = tf.layers.dense(seq, self.num_factor, activation=None)
                V = tf.layers.dense(seq, self.num_factor, activation=None)

                # Split and concat
                Q_ = tf.concat(tf.split(Q, self.num_head, axis=2), axis=0)
                K_ = tf.concat(tf.split(K, self.num_head, axis=2), axis=0)
                V_ = tf.concat(tf.split(V, self.num_head, axis=2), axis=0)

                # Multiplication
                outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

                # Scale
                outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

                # Key Masking
                key_masks = tf.sign(tf.reduce_sum(tf.abs(seq), axis=-1))
                key_masks = tf.tile(key_masks, [self.num_head, 1])
                key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(seq_avg_norm)[1], 1])

                paddings = tf.ones_like(outputs)*(-2**32+1)
                outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

                # Causality (Future blinding)
                diag_vals = tf.ones_like(outputs[0, :, :])
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
                #tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # for earlier tf versions
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

                paddings = tf.ones_like(masks)*(-2**32+1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

                # Activation
                outputs = tf.nn.softmax(outputs)

                # Query Masking
                query_masks = tf.sign(tf.reduce_sum(tf.abs(seq_avg_norm), axis=-1))
                query_masks = tf.tile(query_masks, [self.num_head, 1])
                query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(seq)[1]])
                outputs *= query_masks

                # Dropouts
                outputs = tf.nn.dropout(outputs, keep_prob=1-self.dropout_rate)

                # Weighted sum
                outputs = tf.matmul(outputs, V_)

                # Restore shape
                outputs = tf.concat(tf.split(outputs, self.num_head, axis=0), axis=2)

                # Residual connection
                outputs += seq_avg_norm

                # Layer normalization
                outputs = self.layer_normalize(outputs)

                # Feed forward
                # Layer 1
                outputs_ = tf.layers.dense(outputs, self.num_factor, activation=tf.nn.relu, use_bias=True)
                outputs_ = tf.nn.dropout(outputs_, keep_prob=1-self.dropout_rate)

                # Layer 2
                outputs_ = tf.layers.dense(outputs_, self.num_factor, activation=None, use_bias=True)
                outputs_ = tf.nn.dropout(outputs_, keep_prob=1-self.dropout_rate)

                # Residual connection
                outputs += outputs_

                seq_emb_long = outputs * self.mask

            seq_emb_long = self.layer_normalize(seq_emb_long)

        with tf.name_scope("train"):
            logits_short = tf.matmul(tf.reshape(seq_emb_short, [tf.shape(self.input_seq)[0] * self.max_len, self.num_factor]), tf.transpose(item_embedding))
            logits_long = tf.matmul(tf.reshape(seq_emb_long, [tf.shape(self.input_seq)[0] * self.max_len, self.num_factor]), tf.transpose(item_embedding))
            logits_full = args.alpha * logits_short + (1 - args.alpha) * logits_long
            target = tf.reshape(tf.cast(tf.not_equal(self.pred_seq, self.num_item), tf.float32), [tf.shape(self.input_seq)[0] * self.max_len]) # [batch_size * max_len]
            self.loss_rec = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_full, labels=tf.reshape(self.pred_seq, [-1])) * target) / tf.reduce_sum(target)

            item_tail_norm = self.item_tail / (tf.reduce_max(self.item_tail, 1, True) + 1e-24) # [batch_size * max_len, num_item]
            label_kd_softmax = tf.nn.softmax(item_tail_norm / args.tau, 1)
            item_head_emb = tf.nn.dropout(tf.nn.embedding_lookup(item_embedding_, self.item_head) * (self.num_factor ** 0.5), keep_prob=1-self.dropout_rate)
            logits_kd = tf.matmul(item_head_emb, tf.transpose(item_embedding_))
            logits_kd_softmax = tf.nn.softmax(logits_kd / args.tau, 1) # [batch_size * max_len, num_item+1]
            self.loss_kd = -tf.reduce_mean(tf.reduce_sum(label_kd_softmax * tf.log(logits_kd_softmax + 1e-24), 1))

            self.loss = self.loss_rec + args.lambda_kd * self.loss_kd + self.l2_reg * tf.reduce_sum([tf.nn.l2_loss(va) for va in tf.trainable_variables()])
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.name_scope("test"):
            self.input_u_mask = tf.placeholder(tf.float32, [None, self.num_item], name="input_u_mask")
            self.test_logits = tf.reshape(logits_full, [tf.shape(self.input_seq)[0], self.max_len, self.num_item + 1])[:, -1, :-1] - 2 ** 32 * self.input_u_mask

    def layer_normalize(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(self.num_factor))
        gamma = tf.Variable(tf.ones(self.num_factor))
        normalized = (inputs - mean) / ((variance + 1e-24) ** 0.5)
        outputs = gamma * normalized + beta
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="MQSA_TED")
    parser.add_argument("--dataset", type=str, default="beauty") # ['yelp', 'sports', 'beauty', 'toys']
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
