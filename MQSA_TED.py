# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


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
            logits_kd_softmax = tf.nn.softmax(logits_kd / args.tau, 1)  # [batch_size * max_len, num_item+1]
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
