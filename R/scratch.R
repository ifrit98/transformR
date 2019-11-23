
# Vanilla without scale weight
compute_luong_score <-
  function(query, keys, query_depth, scale = NULL, return_context = TRUE) {
    processed_query <-
      layer_dense(query, units = query_depth, use_bias = FALSE) %>%
      tf$expand_dims(axis = 1L)

    score <-
      tf$matmul(processed_query, keys, transpose_b = TRUE) %>%
      tf$transpose(list(0L, 2L, 1L))

    alignments <-
      tf$nn$softmax(if (is.null(scale)) score else scale * score)


    if (return_context) {
      context <-
        tf$keras$backend$sum(keys * alignments,
                             axis = -1L,
                             keepdims = FALSE)
      return(list(alignments, context))
    }

    alignments
  }

# Vanilla without norm
compute_bahdanau_score <-
  function(query, keys, query_depth, attention_v = NULL) {

    if (is.null(attention_v))
      attention_v <- tf$Variable(tf$random$normal(shape = list(query_depth)))
    processed_query <-
      layer_dense(query, units = query_depth, use_bias = FALSE) %>%
      tf$expand_dims(1L)

    scores <-
      tf$reduce_sum(attention_v * tf$tanh(keys + processed_query), list(2L))

    scores
  }


batch <- 16L
units <- 128L # units == query depth == RNN encoder out units
max_time <- 256L # Sequence length

# Batch = 16, Timesteps = 256, Hidden units = 128
# (1, 128) is a hidden state at time = t.

# query == decoder hidden state at time t (batch, units)
query <- tf$random$normal(shape = list(batch, units))
# keys == encoder hiddens states (i.e. return sequences) (batch, maxtime, units)
keys  <- tf$random$normal(shape = list(batch, max_time, units))

scores <- compute_bahdanau_score(query, keys, units)
scores <- compute_luong_score(query, keys, units)

alignments <- tf$nn$softmax(scores)



context <-
  tf$keras$backend$sum(query * alignments, axis = -1L, keepdims = TRUE)

# Then take context vector and concatenate with hidden state of decoder
# SHAPES of DECODER OUT and CONTEXT INCORRECT??
decoder_out <- tf$random$normal(shape = list(batch, units))
ff_input <- tf$concat(list(context, decoder_out), axis = 1L)

# Pass through a ff layer, the output of which indicates output word of
# this the current time step.
## FF no of output units/activation type?
output <- layer_dense(ff_input, units = 1L, activation = 'sigmoid')





# num_heads <- 4L
# x <- tf$random$normal(list(16L, 8192L, 64L))
# c(q,k,v) %<-% compute_qkv(x)
# q <- split_heads(q, num_heads)
# k <- split_heads(k, num_heads)
# v <- split_heads(v, num_heads)
# y <- local_attention_1d(q, k, v)
