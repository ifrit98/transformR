#' Multiplicative attention (Luong)
#'
#' Accepts set of hidden states from encoder and concatenates
#' them together, then scoring them with decoder hidden state
#' and apply softmax before multiplying each vector by its
#' softmax score and summing to achieve context vector for
#' timestep t.
MultiplicativeAttention <-
  R6::R6Class(
    "BahdanauAttention",

    inherit = KerasLayer,

    public = list(
      encoder_hidden_states = NULL,

      initialize = function(encoder_hidden_states) {
        self$encoder_hidden_states <- encoder_hidden_states

      },

      build = function(input_shape) {

      },

      call = function(x, mask = NULL) {

      }

    )
  )

batch <- 16L
units <- 128L # units == query depth
max_time <- 256L # Sequence length

# Vanilla without scale weight
compute_luong_score <- function(query, keys, query_depth, scale = NULL) {

  processed_query <-
    layer_dense(query, units = query_depth, use_bias = FALSE) %>%
    tf$expand_dims(axis = 1L)

  score <-
    tf$matmul(processed_query, keys, transpose_b = TRUE) %>%
    tf$squeeze(axis = 1L)

  if (!is.null(scale)) return(scale * score)

  score
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


# Batch = 16, Timesteps = 256, Hidden units = 128
# (1, 128) is a hidden state at time = 1.
query <- tf$random$normal(shape = list(batch, max_time))
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


#' Additive attention (Bahdanau)
#'
#' Accepts set of hidden states from encoder and concatenates
#' them together, then scoring them with decoder hidden state
#' and apply softmax before multiplying each vector by its
#' softmax score and summing to achieve context vector for
#' timestep t.
AdditiveAttention <-
  R6::R6Class(
    "BahdanauAttention",

    inherit = KerasLayer,

    public = list(
      encoder_hidden_states = NULL,

      initialize = function(encoder_hidden_states) {
        self$encoder_hidden_states <- encoder_hidden_states

      },

      build = function(input_shape) {

      },

      call = function(x, mask = NULL) {

      }

    )
  )



SelfAttention <-
  R6::R6Class(
    "SelfAttention",

    inherit = KerasLayer,

    public = list(

      Wq = NULL,
      Wk = NULL,
      Wv = NULL,
      kernel_initializer = NULL,
      kernel_regularizer = NULL,

      initialize = function() {},

      build = function(input_shape) {
        self$Wq <- self$add_weight(
          name = 'Wq',
          shape = self$kernel_shape,
          initializer = self$kernel_initializer,
          regularizer = self$kernel_regularizer,
          trainable = TRUE
        )
        self$Wk <- self$add_weight(
          name = 'Wk',
          shape = self$kernel_shape,
          initializer = self$kernel_initializer,
          regularizer = self$kernel_regularizer,
          trainable = TRUE
        )
        self$Wv <- self$add_weight(
          name = 'Wv',
          shape = self$kernel_shape,
          initializer = self$kernel_initializer,
          regularizer = self$kernel_regularizer,
          trainable = TRUE
        )
      },

      call = function(x, mask = NULL) {
        # # Do argument checking for list or single argument: see t2t code
        # # If only first layer!  Break out embedding before layer
        # x <- self$embed_input(x)
        q <- tf$matmul(x, self$Wq)
        v <- tf$matmul(x, self$Wk)
        k <- tf$matmul(x, self$Wv)

        dk <- shape_list2(k)[[3]] # channel dim

        qk <- tf$divide(tf$matmul(q, tf$transpose(k)),
                        tf$sqrt(dk))

        soft <- layer_dense(qk, activation = 'softmax')

        layer_multiply(list(soft, v))
      },

      comput_output_shape = function() {},

      embed_input = function() {},
    )
  )


layer_self_attention <-
  function(object,
           name = NULL,
           trainable = TRUE) {
    create_layer(SelfAttention,
                 object,
                 list(
                   name = name,
                   trainable = trainable
                 ))
  }
