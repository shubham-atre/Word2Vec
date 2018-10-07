import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    A = log(exp({u_o}^T v_c))
	B = log(\sum{exp({u_w}^T v_c)})

    ==========================================================================
    """
    u_dot_v = tf.multiply(true_w, inputs)
    A = tf.reduce_sum(u_dot_v, axis=1, keep_dims=True)

    uw_vc = tf.matmul(inputs, true_w, transpose_b=True)
    sum = tf.reduce_sum(tf.exp(uw_vc), axis=1, keep_dims=True)
    B = tf.log(sum)

    rs = tf.subtract(B, A)

    return rs

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):

    embedding_size = inputs.get_shape()[1]

    uc = inputs

    uo = tf.nn.embedding_lookup(weights, labels)
    uo = tf.reshape(uo, [-1, embedding_size])

    sample_tensor = tf.convert_to_tensor(sample)
    ux = tf.nn.embedding_lookup(weights, sample_tensor)

    bo = tf.nn.embedding_lookup(biases, labels)

    bx = tf.nn.embedding_lookup(biases, sample_tensor)
    bx = tf.reshape(bx, [sample.shape[0], 1])

    k = sample.shape[0]

    pr_wo = tf.gather(unigram_prob, labels)

    pr_wx = tf.gather(unigram_prob, sample)
    pr_wx = tf.reshape(pr_wx, [k,1])

    uc_uo = tf.reduce_sum(tf.multiply(uc, uo), axis=1, keep_dims=True)

    s_wc_wo = tf.add(uc_uo, bo)

    log_kPr_wo = tf.log(tf.add(tf.multiply(pr_wo, k), 1e-10))

    sig_o = tf.sigmoid(tf.subtract(s_wc_wo, log_kPr_wo))

    log_sigma_o = tf.log(tf.add(sig_o, 1e-10))

    uc_ux = tf.matmul(uc, ux, transpose_b=True)

    s_wc_wx = tf.add(uc_ux, tf.transpose(bx))

    log_kPr_wx = tf.log(tf.add(tf.multiply(pr_wx, k), 1e-10))

    sig_x = tf.sigmoid(tf.subtract(s_wc_wx, tf.transpose(log_kPr_wx)))

    log_sigma_x = tf.reduce_sum(tf.log(tf.add(tf.subtract(1.0, sig_x), 1e-10)), axis=1, keep_dims=True)

    rs = tf.add(log_sigma_o, log_sigma_x)

    return tf.multiply(rs, -1)

    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    ==========================================================================
    """
