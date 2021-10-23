from __future__ import print_function
import os

from flip_gradient import flip_gradient
from amazon_utils import *
import argparse
from tensorflow import set_random_seed
from logger import get_logger
import datetime


INPUT_DIM = 5000
NUM_CLASS = 2


class OfficeModel:

    def __init__(self, input_feature_size, z_hidden_size):
        self.input_shape = input_feature_size
        self.d_input_size = 2
        self.d_hidden_size = 1
        self.z_hidden_size = z_hidden_size
        pass

    def inference(self, x, d, is_reuse, l, is_training, dropout_rate):
        with tf.variable_scope("inference", reuse=is_reuse):
            domain_code = self.domain_encoder(d=d, is_reuse=is_reuse)
            h_feature_exact_1 = self.feature_encoder(x=x, is_training=is_training,
                                                     domain_code=domain_code, dropout_rate=dropout_rate)

            label_logits = self.label_prediction(feature=h_feature_exact_1, is_reuse=is_reuse, is_training=is_training,
                                                 dropout_rate=dropout_rate)

            rec = self.decoder(latent_variable=h_feature_exact_1, domain_code=domain_code, is_reuse=is_reuse,
                               is_training=is_training, dropout_rate=dropout_rate)

        domain_logits = self.domain_prediction(feature=h_feature_exact_1, is_reuse=is_reuse, l=l)

        return label_logits, domain_logits, rec, h_feature_exact_1

    def inference_trans(self, x, is_reuse, l, is_training, domain, dropout_rate, domain_trans, trans_is_reuse):
        label_logits, domain_logits, rec, inner_code = \
            self.inference(x=x, is_reuse=is_reuse, l=l, is_training=is_training,
                           d=domain, dropout_rate=dropout_rate)

        with tf.variable_scope("inference", reuse=is_reuse):
            domain_code_trans = self.domain_encoder(d=domain_trans, is_reuse=True)
            rec_trans = self.decoder(latent_variable=inner_code, domain_code=domain_code_trans, is_reuse=True,
                                     is_training=is_training, dropout_rate=dropout_rate)

        with tf.variable_scope("trans", reuse=trans_is_reuse):
            rec_label_trans = \
                self.label_predictor_regularization(feature=rec_trans, is_reuse=trans_is_reuse, is_training=is_training,
                                                    dropout_rate=dropout_rate)

        return label_logits, domain_logits, rec, inner_code, rec_label_trans

    def label_predictor_regularization(self, feature, is_reuse, is_training, dropout_rate,
                                       label_predictor_hidden_state=3000):
        with tf.variable_scope("label_predictor_trans", reuse=is_reuse):
            W_feature_exact_0 = weight_variable(shape=[self.input_shape, label_predictor_hidden_state],
                                                name="feature_extractor_weight_0")
            b_feature_exact_0 = bias_variable(shape=[label_predictor_hidden_state], name="feature_extractor_biases_0")
            h_feature_exact_0 = tf.nn.relu(tf.matmul(feature, W_feature_exact_0) + b_feature_exact_0)
            h_feature_exact_0 = tf.layers.dropout(h_feature_exact_0, training=is_training, rate=dropout_rate)

            W_feature_exact_1 = weight_variable(shape=[label_predictor_hidden_state, self.z_hidden_size],
                                                name="feature_extractor_weight_1")
            b_feature_exact_1 = bias_variable(shape=[self.z_hidden_size], name="feature_extract_biases_1")
            h_feature_exact_1 = tf.nn.relu(tf.matmul(h_feature_exact_0, W_feature_exact_1) + b_feature_exact_1)
            h_feature_exact_1 = tf.layers.dropout(h_feature_exact_1, training=is_training, rate=dropout_rate)

            W_fc0 = weight_variable(shape=[self.z_hidden_size, NUM_CLASS], name="fc0_w")
            b_fc0 = bias_variable(shape=[NUM_CLASS], name="fc0_b")
            label_logits = tf.matmul(h_feature_exact_1, W_fc0) + b_fc0

        return label_logits

    def decoder(self, latent_variable, domain_code, is_reuse, is_training, dropout_rate):
        with tf.variable_scope("feature_decoder", reuse=is_reuse):
            feature = tf.concat([latent_variable, domain_code], axis=-1)
            decoder_w0 = weight_variable(shape=[self.z_hidden_size + self.d_hidden_size, self.input_shape],
                                         name="decoder_weight_0")
            decoder_b0 = bias_variable(shape=[self.input_shape], name="decoder_biases_0")
            decoder_h0 = tf.matmul(feature, decoder_w0) + decoder_b0
            decoder_h0 = tf.nn.relu(decoder_h0)
            decoder_h0 = tf.layers.dropout(decoder_h0, training=is_training, rate=dropout_rate)
            # decoder_h0 = tf.contrib.layers.layer_norm(decoder_h0, scale=True)
            # decoder_h0 = tf.contrib.layers.batch_norm(decoder_h0, scale=True)

        return decoder_h0

    def domain_encoder(self, d, is_reuse):
        with tf.variable_scope("domain_encoder", reuse=is_reuse):
            W_d_feature_extract_0 = weight_variable(shape=[self.d_input_size, self.d_hidden_size],
                                                    name="doamin_feature_extractor_weight_1")
            b_d_feature_extract_0 = bias_variable(shape=[self.d_hidden_size], name="domain_feature__extract_biases_1")

            d_feature_extract_0 = tf.matmul(d, W_d_feature_extract_0) + b_d_feature_extract_0

        return d_feature_extract_0

    def feature_encoder(self, x, is_training, domain_code, dropout_rate):
        x = tf.concat([x, domain_code], axis=1)

        W_feature_exact_0 = weight_variable(shape=[self.input_shape + self.d_hidden_size, 4000],
                                            name="feature_extractor_weight_1")
        b_feature_exact_0 = bias_variable(shape=[4000], name="feature_extractor_biases_1")
        h_feature_exact_0 = tf.nn.relu(tf.matmul(x, W_feature_exact_0) + b_feature_exact_0)
        h_feature_exact_0 = tf.layers.dropout(h_feature_exact_0, training=is_training, rate=dropout_rate)

        W_feature_exact_1 = weight_variable(shape=[4000, self.z_hidden_size], name="feature_extractor_weight_2")
        b_feature_exact_1 = bias_variable(shape=[self.z_hidden_size], name="feature_extract_biases_2")
        h_feature_exact_1 = tf.nn.relu(tf.matmul(h_feature_exact_0, W_feature_exact_1) + b_feature_exact_1)
        h_feature_exact_1 = tf.layers.dropout(h_feature_exact_1, training=is_training, rate=dropout_rate)

        return h_feature_exact_1

    def label_prediction(self, feature, is_reuse, is_training, dropout_rate):
        with tf.variable_scope('label_predictor', reuse=is_reuse):
            W_fc0 = weight_variable(shape=[self.z_hidden_size, 100], name="fc0_w")
            b_fc0 = bias_variable(shape=[100], name="fc0_b")
            h_fc0 = tf.matmul(feature, W_fc0) + b_fc0
            h_fc0 = tf.layers.dropout(h_fc0, training=is_training, rate=dropout_rate)

            W_fc1 = weight_variable(shape=[100, NUM_CLASS], name="fc1_w")
            b_fc1 = bias_variable(shape=[NUM_CLASS], name="f1_b")
            label_logits = tf.matmul(h_fc0, W_fc1) + b_fc1
        return label_logits

    def domain_prediction(self, feature, is_reuse, l):
        with tf.variable_scope("domain_predictor", reuse=is_reuse):
            feat = flip_gradient(feature, l)

            d_W_fc0 = weight_variable(shape=[self.z_hidden_size, 100], name="fcd_w0")
            d_b_fc0 = bias_variable(shape=[100], name="fcd_b0")
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

            d_W_fc1 = weight_variable(shape=[100, 2], name="fcd_w1")
            d_b_fc1 = bias_variable(shape=[2], name="fcd_b1")
            d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

        return d_logits


def get_one_hot_label(y):
    n_values = np.max(y) + 1
    one_hot_label = np.eye(n_values)[y]
    return one_hot_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/data/", type=str)
    parser.add_argument("--result", default="result", type=str)
    parser.add_argument("--src", default="kitchen", type=str)
    parser.add_argument("--tgt", default="electronics", type=str) # books, electronics, dvd, kitchen
    parser.add_argument("--z_hidden_size", default=4096, type=int)
    parser.add_argument("--dropout_rate", default=0.75, type=float)
    parser.add_argument("--seed", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--l2_loss_weight", default=1e-5, type=float)
    parser.add_argument("--rec_loss_weight", default=1.0, type=float)
    parser.add_argument("--trans_class_weight", default=0.0005, type=float)

    args = parser.parse_args()

    cur_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    logger = get_logger(os.path.join(args.result, "current_time_%s.log" % cur_time))
    logger.info(args)

    data_dir = args.data_dir
    source_name = args.src
    target_name = args.tgt

    source_train_input, source_train_label, \
    target_train_input, target_train_label, \
    target_test_input, target_test_label = load_amazon(
        source_name=source_name, target_name=target_name, data_folder=data_dir)

    source_train_y = get_one_hot_label(source_train_label)
    target_train_y = get_one_hot_label(target_train_label)
    target_test_y = get_one_hot_label(target_test_label)

    print(source_train_input.shape)
    print(source_train_label.shape)
    print(target_train_input.shape)
    print(target_train_label.shape)
    print(target_test_input.shape)
    print(target_test_label.shape)

    batch_size = 128
    num_steps = 10000
    set_random_seed(1)
    graph = tf.get_default_graph()
    with graph.as_default():
        model = OfficeModel(input_feature_size=INPUT_DIM, z_hidden_size=args.z_hidden_size)

        source_input = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_DIM])
        source_label = tf.placeholder(dtype=tf.int32, shape=[None, NUM_CLASS])
        target_input = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_DIM])
        target_label = tf.placeholder(dtype=tf.int32, shape=[None, NUM_CLASS])

        source_domain = tf.placeholder(tf.float32, [None, 2])
        target_domain = tf.placeholder(tf.float32, [None, 2])
        domain_trans = tf.placeholder(tf.float32, [None, 2])
        learning_rate = tf.placeholder(tf.float32, [])

        alpha = tf.placeholder(tf.float32, [])
        train_mode = tf.placeholder(tf.bool, [])

        source_label_logits, source_domain_logits, source_rec, source_inner_code = \
            model.inference(x=source_input, is_reuse=False, l=alpha,
                            is_training=train_mode, d=source_domain, dropout_rate=args.dropout_rate)

        target_label_logits, target_domain_logits, target_rec, target_inner_code = \
            model.inference(x=target_input, is_reuse=True, l=alpha,
                            is_training=train_mode, d=target_domain, dropout_rate=args.dropout_rate)

        source_trans_label_logits, source_trans_domain_logits, source_trans_rec, \
        source_trans_inner_code, rec_label_trans = \
            model.inference_trans(x=source_input, is_reuse=True, is_training=train_mode, domain=source_domain,
                                  dropout_rate=args.dropout_rate, l=alpha, domain_trans=domain_trans,
                                  trans_is_reuse=False)

        source_cis_label_logits, source_cis_domain_logits, source_cis_rec, source_cis_inner_code, rec_label_cis = \
            model.inference_trans(x=source_input, is_reuse=True, is_training=train_mode, domain=source_domain,
                                  dropout_rate=args.dropout_rate, l=alpha, domain_trans=source_domain,
                                  trans_is_reuse=True)

        trans_class_loss = \
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=rec_label_cis,
                                                                      labels=source_label)) + \
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=rec_label_trans,
                                                                      labels=source_label))

        src_label_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=source_label_logits, labels=source_label))

        rec_logits = tf.concat([source_rec, target_rec], axis=0)
        data_input = tf.concat([source_input, target_input], axis=0)
        rec_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(rec_logits - data_input), axis=-1)))

        label_loss = src_label_loss

        domain_logits = tf.concat([source_domain_logits, target_domain_logits], 0)
        source_target_domain_label = tf.concat([source_domain, target_domain], 0)

        domain_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=domain_logits, labels=source_target_domain_label))

        infer_var_list = [v for v in tf.trainable_variables() if v.name.split("/")[0] == "inference"]
        # infer_var_list = [v for v in tf.trainable_variables()]

        greg_loss = 1e-5 * tf.reduce_mean([tf.nn.l2_loss(x) for x in infer_var_list if "w" in x.name])
        # greg_loss = 5e-5 * tf.reduce_mean([tf.nn.l2_loss(x) for x in infer_var_list])

        total_loss = label_loss + domain_loss + greg_loss + 1 * rec_loss + 0.005 * trans_class_loss
        # total_loss = label_loss + domain_loss + greg_loss + 0.001 * rec_loss + 0.001 * trans_class_loss

        # dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.3).minimize(total_loss)
        dann_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(total_loss)

        target_correct_label = tf.equal(tf.argmax(target_label, 1), tf.argmax(target_label_logits, 1))
        target_label_acc = tf.reduce_mean(tf.cast(target_correct_label, tf.float32))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as session:
            tf.global_variables_initializer().run()

            gen_source_batch = batch_generator([source_train_input, source_train_y], batch_size)
            gen_target_batch = batch_generator([target_train_input, target_train_y], batch_size)
            # gen_source_batch = balance_batch_generator([source_train_input, source_train_y])
            # gen_target_batch = balance_batch_generator([target_train_input, target_train_y])

            source_domain_input = np.tile([1., 0.], [batch_size, 1])
            target_domain_input = np.tile([0., 1.], [batch_size, 1])

            best_result = 0.0
            

            tf_board_data_size = 0.0
            tf_board_total_loss = 0.0
            tf_board_label_loss = 0.0
            tf_board_rec_loss = 0.0
            tf_board_trans_class_loss = 0.0
            tf_board_domain_loss = 0.0

            for global_steps in range(num_steps):
                p = float(global_steps) / num_steps
                # L = min(2 / (1. + np.exp(-10. * p)) - 1, 1.0)
                # L = 2 / (1. + np.exp(-10. * p)) - 1
                L = 0.15
                # lr = max(lr / (1. + 10 * p) ** 0.55, 0.0001)

                X0, y0 = next(gen_source_batch)
                X1, y1 = next(gen_target_batch)
                bs = len(X0)
                _, batch_total_loss, batch_domain_loss, batch_label_loss, batch_rec_loss, batch_trans_class_loss = session.run(
                    [dann_train_op, total_loss, domain_loss, label_loss, rec_loss, trans_class_loss],
                    feed_dict={source_input: X0, source_label: y0, target_input: X1, learning_rate: args.lr, alpha: L,
                               source_domain: source_domain_input, target_domain: target_domain_input,
                               train_mode: True, domain_trans: target_domain_input})

                tf_board_data_size += bs
                tf_board_total_loss += batch_total_loss * bs
                tf_board_label_loss += batch_label_loss * bs
                tf_board_rec_loss += batch_rec_loss * bs
                tf_board_trans_class_loss += batch_trans_class_loss * bs
                tf_board_domain_loss += batch_domain_loss * bs

                if global_steps % 100 == 0:

                    test_target_domain_input = np.tile([0., 1.], [len(target_test_input), 1])
                    target_label_accuracy = session.run(target_label_acc,
                                                        feed_dict={target_input: target_test_input,
                                                                   target_label: target_test_y,
                                                                   train_mode: False,
                                                                   target_domain: test_target_domain_input})
                    if target_label_accuracy > best_result:
                        best_result = target_label_accuracy


                    # print("global_step:%d, batch_total_loss:%f\t, batch_domain_loss:%f\t, batch_label_loss:%f, \n"
                    #       "target_label_accuracy:%f, lr:%f, L:%f, best result:%f" % (
                    #           global_steps, batch_total_loss, batch_domain_loss, batch_label_loss,
                    #           target_label_accuracy, lr, L, best_result))
                    # print("batch_rec_loss:", batch_rec_loss)
                    # print("batch_trans_class_loss:", batch_trans_class_loss)

                    # print()
                    logger.info("global_step:%d, batch_total_loss:%f\t, batch_domain_loss:%f\t, batch_label_loss:%f,\t "
                          "batch_rec_loss:%f \t batch_trans_class_loss:%f\n"
                          "target_label_accuracy:%f, lr:%f, L:%f, best result:%f" % (
                              global_steps,
                              tf_board_total_loss / tf_board_data_size,
                              tf_board_domain_loss / tf_board_data_size,
                              tf_board_label_loss / tf_board_data_size,
                              tf_board_rec_loss / tf_board_data_size,
                              tf_board_trans_class_loss / tf_board_data_size,
                              target_label_accuracy, args.lr, L, best_result))

    # record = open("best_result_good.txt", mode="a")
    # print("%s->%s:%f, %d" % (args.src, args.tgt, best_result,args.seed), file=record)
    logger.info("%s->%s:%f, %d" % (args.src, args.tgt, best_result,args.seed))
