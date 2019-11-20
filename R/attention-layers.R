library(tensorflow)
library(keras)

#' Multiplicative attention (Luong et al. 2016)
#'
#' Accepts set of hidden states from encoder and concatenates
#' them together, then scoring them with decoder hidden state
#' and apply softmax before multiplying each vector by its
#' softmax score and summing to achieve context vector for
#' timestep t.
MultiplicativeAttention <-
  R6::R6Class(
    "MultiplicativeAttention",

    inherit = KerasLayer,

    public = list(
      query_depth = NULL,
      return_context = NULL,
      kernel = NULL,
      scale = NULL,
      use_scale = NULL,

      initialize = function(query_depth, return_context, use_scale) {
        self$query_depth <- query_depth
        self$return_context <- return_context
        self$use_scale <- use_scale

      },

      build = function(input_shape) {
        stopifnot(mode(input_shape) == "list")

        self$kernel <- self$add_weight(
          'kernel',
          shape = list(input_shape[[1]][[1]], self$query_depth),
          initializer = "glorot_normal"
        )

        self$scale <- self$add_weight(
          'scale',
          shape = list(),
          initializer = 'ones'
        )
      },

      call = function(x, mask = NULL) {
        stopifnot(mode(x) == "list")

        c(query, keys) %<-% x

        processed_query <-
          tf$matmul(query, self$kernel) %>%
          tf$expand_dims(axis = 1L)

        score <-
          tf$matmul(processed_query, keys, transpose_b = TRUE) %>%
          tf$transpose(list(0L, 2L, 1L))

        alignments <-
          tf$nn$softmax(if (is.null(scale)) score else self$scale * score)

        if (self$return_context) {
          context <-
            tf$keras$backend$sum(keys * alignments,
                                 axis = -1L,
                                 keepdims = FALSE)
          return(list(alignments, context))
        }

        alignments
      }

    )
  )


#' Layer wrapper for Luong (multiplicative) attention.
#'
#' Takes a list() of (query, key) tensor of shape (batch, hidden_units),
#' (batch, maxtime, hidden_units) and returns a context vector for the
#' input sequences.  Query is usually the current hidden state of a
#' decoder and key is usually output of an RNN encoder (hidden states).
#'
#' @export
#' @example
#' batch <- 16L
#' units <- 128L # units == query depth == RNN encoder out units
#' max_time <- 256L # Sequence length
#' # query == decoder hidden state at time t (batch, units)
#' query <- tf$random$normal(shape = list(batch, units))
#' # key == encoder hiddens states (i.e. return sequences) (batch, maxtime, units)
#' key <- tf$random$normal(shape = list(batch, max_time, units))
#' context <- layer_multiplicative_attention(list(query, key), key$get_shape()[-1])
layer_multiplicative_attention <-
  function(object,
           query_depth,
           return_context = TRUE,
           use_scale = TRUE,
           name = NULL,
           trainable = TRUE) {
    create_layer(MultiplicativeAttention,
                 object,
                 list(
                   query_depth = as.integer(query_depth),
                   return_context = return_context,
                   use_scale = use_scale
                 ))
  }



#' Additive attention (Bahdanau et al. 2015)
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
      query_depth = NULL,
      attention_v = NULL,
      kernel = NULL,
      return_context = NULL,

      initialize = function(query_depth, return_context) {
        self$query_depth <- query_depth
        self$return_context <- return_context
      },

      build = function(input_shape) {
        query_dims <- input_shape[[1]]
        keys_dims  <- input_shape[[2]]

        self$attention_v <-
          tf$Variable(tf$random$normal(shape = list(self$query_depth)))

        self$kernel <- self$add_weight(
          'kernel',
          shape = list(query_dims[[1]], self$query_depth),
          initializer = "glorot_normal"
        )

      },

      call = function(x, mask = NULL) {
        stopifnot(mode(x) == "list")

        c(query, keys) %<-% x

        if (!self$query_depth == keys$get_shape()[2])
          stop(
            paste(
              "Query projection units must equal number of",
              "dimensions of keys tensor. Got:",
              self$query_depth,
              "and",
              keys$get_shape()[2],
              "Perhaps you need to set query_depth to the keys' last dimension"
            )
          )

        processed_query <-
          tf$matmul(query, self$kernel) %>%
          tf$expand_dims(1L)

        scores <-
          tf$reduce_sum(self$attention_v * tf$tanh(keys + processed_query),
                        list(2L))

        alignments <- tf$nn$softmax(scores)

        if (self$return_context) {
          context <-
            tf$keras$backend$sum(keys * tf$expand_dims(alignments, 2L),
                                 axis = -1L,
                                 keepdims = TRUE)
          return(list(alignments, context))
        }

        alignments
      }

    )
  )


#' Layer wrapper for Bahdanau (additive) attention.
#'
#' Takes a list() of (query, key) tensor of shape (batch, hidden_units),
#' (batch, maxtime, hidden_units) and returns a context vector for the
#' input sequences.  Query is usually the current hidden state of a
#' decoder and key is usually output of an RNN encoder (hidden states).
#'
#' @export
#' @example
#' batch <- 16L
#' units <- 128L # units == query depth == RNN encoder out units
#' max_time <- 256L # Sequence length
#' # query == decoder hidden state at time t (batch, units)
#' query <- tf$random$normal(shape = list(batch, units))
#' # key == encoder hiddens states (i.e. return sequences) (batch, maxtime, units)
#' key <- tf$random$normal(shape = list(batch, max_time, units))
#' context <- layer_additive_attention(list(query, key), key$get_shape()[-1])
layer_additive_attention <-
  function(object,
    query_depth,
    return_context = TRUE,
    name = NULL,
    trainable = TRUE) {
    create_layer(
      AdditiveAttention,
      object,
      list(
        query_depth = as.integer(query_depth),
        return_context = return_context,
        name = name,
        trainable = trainable
      )
    )
  }



#' Self attention layer, implementing scaled dot product attention
#' Vaswani et al. 2017
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
