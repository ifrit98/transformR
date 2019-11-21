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


        
#' R6 class containing local self attention logic
LocalSelfAttention <- R6::R6Class(
  "LocalSelfAttention",
  
  inherit = KerasLayer,
  
  public = list(
    
    units = NULL,
    attention_width = NULL,
    attention_type = NULL,
    use_attention_bias = NULL,
    kernel_initializer = NULL,
    kernel_regularizer = NULL,
    bias_initializer = NULL,
    bias_regularizer = NULL,
    Wt = NULL,
    Wx = NULL,
    Wa = NULL,
    bh = NULL,
    ba = NULL,
    
    initialize = function(units,
                          attention_width,
                          attention_type,
                          use_attention_bias,
                          kernel_initializer,
                          kernel_regularizer,
                          bias_initializer,
                          bias_regularizer) {
      self$units              <- units
      self$attention_width    <- attention_width
      self$attention_type     <- attention_type
      self$use_attention_bias <- use_attention_bias
      self$kernel_initializer <- kernel_initializer
      self$kernel_regularizer <- kernel_regularizer
      self$bias_initializer   <- bias_initializer
      self$bias_regularizer   <- bias_regularizer
    },
    
    build_additive_attention = function(channels) {
      self$Wt <- self$add_weight(
        shape = list(channels, self$units),
        initializer = self$kernel_initializer,
        regularizer = self$kernel_regularizer,
        name = "Wt"
      )
      
      self$Wx <- self$add_weight(
        shape = list(channels, self$units),
        initializer = self$kernel_initializer,
        name = "Wx"
      )
      
      self$Wa <- self$add_weight(
        shape = list(self$units, 1L),
        initializer = self$kernel_initializer,
        name = "Wa"
      )
      
      if (self$use_attention_bias) {
        self$bh <- self$add_weight(
          shape = list(self$units),
          initializer = self$bias_initializer,
          name = "bh"
        )
      
        self$ba <- self$add_weight(
          shape = list(1L),
          initializer = self$bias_initializer,
          name = "ba"
        )
      }
    },
    
    build_multiplicative_attention = function(channels) {
      self$Wa <- self$add_weight(
        shape = list(channels, channels),
        initializer = self$kernel_initializer,
        name = "Wa"
      )
      
      if (self$use_attention_bias)      
        self$ba <- self$add_weight(
          shape = list(1L),
          initializer = self$bias_initializer,
          name = "ba"
        )
    },
    
    build = function(input_shape) {
      channels <- input_shape[[length(input_shape)]]
      
      if (!self$attention_type %in% c("additive", "multiplicitive"))
        stop("attention_type must be one of: 'additive', 'multiplicative'")
      
      if (self$attention_type == "additive")
        self$build_additive_attention(channels)
      else
        self$build_multiplicative_attention(channels)
      
    },
    
    call = function(x, mask = NULL) {
      seqlen <- shape_list2(x)[[2]]
      
      score <- 
        if (self$attention_type == "additive")
          self$additive_score(x)
        else 
          self$multiplicative_score(x)
      
      # Localize
      lower <- 
        tf$range(0L, seqlen) - as.integer(self$attention_width / 2L) %>% 
        tf$expand_dims(-1L)
      
      upper <- lower + self$attention_width
      indices <- tf$expand_dims(tf$range(0L, seqlen), axis = 0L)

      # Mask out anything wider than attention_width and apply scores
      emission <- 
        score * 
        tf$cast(lower <= indices, tf$float32) * 
        tf$cast(upper > indices, tf$float32) 
  
      sum <- tf$keras$backend$sum(emission, axis = -1L, keepdims = TRUE)
      
      attention <- emission / (sum + tf$keras$backend$epsilon())

      v <- tf$matmul(attention, x)
      
      v
    },
    
    additive_score = function(x) {
      shape  <- shape_list2(x)
      batch  <- shape[[1]]
      seqlen <- shape[[2]]
      
      q <- tf$expand_dims(tf$matmul(x, self$Wt), 2L)
      k <- tf$expand_dims(tf$matmul(x, self$Wx), 1L)
      
      h <- tf$tanh(q + k + if (!is.null(self$bh)) self$bh else tf$constant(0L))

      e <- tf$reshape(tf$matmul(h, self$Wa) + 
                        if (!is.null(self$ba)) self$ba
                        else tf$constant(0L), 
                      list(batch, seqlen, seqlen))
      e
    },
    
    multiplicative_score = function(x) {
      browser()
      score <- tf$keras$backend$batch_dot(
        tf$matmul(x, self$Wa),
        tf$transpose(x, perm = list(0L, 2L, 1L))
      )
      
      if (!is.null(self$ba))
        score <- score + self$ba
      
      score
    },
    
    compute_output_shape = function(input_shape) {
      output_shape <- input_shape
      if (self$return_attention)
        output_shape <- list(output_shape,
                             list(input_shape[[1]],
                                  output_shape[[2]],
                                  input_shape[[2]]))
      output_shape
    }
    
  )
)


      
#' Keras layer wrapper for local self attention.
#' @export
layer_local_self_attention <- function(object,
                                       units,
                                       attention_width,
                                       attention_type = "additive",
                                       use_attention_bias = TRUE,
                                       kernel_initializer = 'glorot_uniform',
                                       kernel_regularizer = NULL,
                                       bias_initializer = 'zeors',
                                       bias_regularizer = NULL) {
  create_layer(
    LocalSelfAttention,
    object,
    list(
      units = as.integer(units),
      attention_width = as.integer(attention_width),
      attention_type = attention_type,
      use_attention_bias = use_attention_bias,
      kernel_initializer = tf$keras$initializers$get(kernel_initializer),
      kernel_regularizer = tf$keras$regularizers$get(kernel_regularizer),
      bias_initializer = tf$keras$initializers$get(bias_initializer),
      bias_regularizer = tf$keras$initializers$get(bias_initializer)
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

        
        
        
# Simple functions to get shapes straight before wrapping in an R6 layer
 
# q: a Tensor with shape [batch, length_q, depth_k]
# k: a Tensor with shape [batch, length_kv, depth_k]
# v: a Tensor with shape [batch, length_kv, depth_v]
dot_product_attention_1d <- function(q, k, v, bias = NULL, dropout = 0) {
  q_shape <- shape_list2(q)
  scalarv <- tf$rsqrt(tf$cast(q_shape[[length(q_shape)]], tf$float32))
  logits  <- tf$matmul(q * scalar, k, transpose_b = TRUE)
  
  if (!is.null(bias))
    logits <- logits + bias
  
  weights <- tf$nn$softmax(logits, name = "attention_weights")
  
  x <- tf$matmul(weights, v)
  
  x
}
      

#' antecedent: Tensor with shape [batch, length, channels]
#' depth: specifying projection layer depth
#' filter_width: how wide should the attention component be
#' padding: must be in: c("valid", "same", "left")
.compute_attention_component <- function(antecedent,
                                         depth,
                                         filter_width = 1L,
                                         padding = 'valid',
                                         name = 'c') {
  layer_lambda(x, function(x) {
    out <- 
      if (filter_width == 1L) 
        layer_dense(x, depth, use_bias = FALSE, name = name)
    else
      layer_conv_1d(x, depth, filter_width, padding = padding, name = name)
    
    out
  }, name = ".compute_attention_component")
}

      
      
#' query  [batch, length_q, channels]
#' memory [batch, length_m, channels] (optional, usually RNN hidden states)
#' return [batch, length, depth] (q, k ,v) tensors      
#' @export
layer_compute_qkv <- function(query,
                              memory = NULL,
                              key_depth = 64L,
                              value_depth = 64L,
                              q_filter_width = 1L,
                              kv_filter_width = 1L,
                              q_padding = 'valid',
                              kv_padding = 'valid') {
  layer_lambda(x, function() {
    if (is.null(memory)) memory <- query
    q <- 
      .compute_attention_component(query, key_depth, q_filter_width, q_padding, "q")
    
    k <- 
      .compute_attention_component(memory, key_depth, kv_filter_width, kv_padding, "k")
    
    v <- 
      .compute_attention_component(memory, key_depth, kv_filter_width, kv_padding, "v")
    
    c(q, k, v)
  }, name = "layer_compute_qkv")
}
      
      
      
      
# TODO: Make this an R6 layer?
#' Input query, key, and value matrices are used to compute dot product
#' attention. (Vaswani et al. 2017)
#' q: a Tensor with shape [batch, length_q, depth_k]
#' k: a Tensor with shape [batch, length_kv, depth_k]
#' v: a Tensor with shape [batch, length_kv, depth_v]      
#' @export
layer_dot_product_attention_1d <- 
  function(q, k, v, bias = NULL, dropout = 0) {
  
  layer_lambda(c(q,k,v), function(x){
    
    c(q,k,v) %<-% x
    
    q_shape <- shape_list2(q)
    
    scalar <- tf$math$rsqrt(tf$cast(q_shape[[length(q_shape)]], tf$float32))
    logits <- tf$matmul(q * scalar, k, transpose_b = TRUE)
    
    if (!is.null(bias))
      logits <- logits + bias
    
    weights <- tf$nn$softmax(logits, name = "attention_weights")
    
    x <- tf$matmul(weights, v)
    
    x
  }, name = "dot_product_attention_1d")
}

      
# TODO: make this an R6 layer?
#' Takes input tensor of shape [batch, seqlen, channels] and
#' creates query, key, and value tensors to pass to attention
#' mechanisms downstream.
#' 
#' query shape [batch, seqlen, filter_depth]
#' key shape   [batch, seqlen, filter_depth]
#' value shape [batch, seqlen, filter_depth]      
#' @export
layer_create_qkv <- 
  function(x, filter_depth, num_parts = 1L, share_kv = FALSE) {
    layer_lambda(x, function(x) {
      x_shape    <- shape_list2(x)
      part_depth <- as.integer(floor(filter_depth / num_parts))
      
      if (!share_kv) {
        combined <- layer_dense(
          x, filter_depth * 3L, use_bias = FALSE, name = "qkv_transform")
      
        c(q, k, v) %<-% tf$split(combined, 3L, axis = 2L)
      }
      else {
        browser()
        q <- layer_dense(
          x, filter_depth, use_bias = FALSE, name = "q_transform")
        
        kv_combined <-
          layer_dense(
            tf$concat(list(x, x), axis = 1L),
            filter_depth,
            use_bias = FALSE,
            name = "kv_transform")
        
        c(k, v) %<-% 
          tf$split(kv_combined, list(x_shape[[2]], x_shape[[2]]), axis = 1L)
      }
      
      q <- q * tf$pow(tf$cast(part_depth, tf$float32), tf$constant(-0.5))
      
      c(q, k, v)
    }, name = "create_qkv")
}
     
      
      
      
#' @export
layer_self_attention_simple <- 
  function(x,
           filter_depth = 32L,
           output_depth = 64L,
           num_parts = 3L,
           dropout = 0,
           share_kv = TRUE) {
    layer_lambda(x, function(x) {
      x_shape <- shape_list2(x)
      
      c(q, k, v) %<-% layer_create_qkv(x, filter_depth, num_parts, share_kv)
      
      bias <- NULL
      x <- layer_dot_product_attention_1d(q, k, v, bias, dropout)
      x <- tf$reshape(x, list(x_shape[[1]], x_shape[[2]], filter_depth))
      x <- layer_dense(x, output_depth, use_bias = FALSE, name = "output_transform")
      
      x
    }, name = "self_attention_simple")
  }
      
      


self_attention <-
  function(x,
           filter_depth = 64L,
           output_depth = 64L,
           num_parts = 3L,
           dropout = 0.2,
           share_kv = TRUE) {
    
  x_shape    <- shape_list2(x)
  part_depth <- as.integer(floor(filter_depth / num_parts))
  
  combined <- layer_dense(
    x, filter_depth * 3L, use_bias = FALSE, name = "qkv_transform")
  
  if (!share_kv)
    c(q, k, v) %<-% tf$split(combined, 3L, axis = 2L)
  else {
    q <- layer_dense(
          x, filter_depth, use_bias = FALSE, name = "q_transform")
    
    kv_combined <-
      layer_dense(
        tf$concat(list(x, x), axis = 1L),
        filter_depth,
        use_bias = FALSE,
        name = "kv_transform")
    
    c(k, v) %<-% 
      tf$split(kv_combined, list(x_shape[[2]], x_shape[[2]]), axis = 1L)
  }
  
  q <- q * tf$pow(tf$cast(part_depth, tf$float32), tf$constant(-0.5))
  
  bias <- NULL
  x <- dot_product_attention_1d(q, k, v, bias, dropout)
  x <- tf$reshape(x, list(x_shape[[1]], x_shape[[2]], filter_depth))
  x <- layer_dense(x, output_depth, use_bias = FALSE, name = "output_transform")
  
  x
}


# batch    <- 16L
# length   <- 256L
# channels <- 128L
# x <- tf$random$normal(shape = list(batch, length, channels))
# y <- self_attention(x)
