


layer_multihead_attention <- function(query,
                                      memory = NULL,
                                      bias = NULL,
                                      key_depth = 64L,
                                      value_depth = 64L,
                                      output_depth = 128L,
                                      num_heads = 4L,
                                      dropout = 0,
                                      attention_type = "dot_product",
                                      q_filter_width = 1L,
                                      kv_filter_width = 1L,
                                      q_padding = "VALID",
                                      kv_padding = "VALID",
                                      max_area_width = 1L,
                                      max_area_height = 1L,
                                      memory_height = 1L,
                                      area_key_mode = "mean",
                                      area_value_mode = "sum",
                                      vars_3d = TRUE) {

  layer_lambda(list(query, memory), function(x) {
    stopifnot(key_depth %% num_heads == 0, value_depth %% num_heads == 0)

    if (typeof(x) == "list" & length(x) > 1) # if (any(grepl("list", class(x))))
      c(query, memory) %<-% x
    else
      query <- x

    vars_3d_num_heads <- if (vars_3d) num_heads else 0

    c(q, k, v) %<-% layer_compute_qkv(query = query,
                                      memory = memory,
                                      key_depth = key_depth,
                                      value_depth = value_depth,
                                      q_filter_width = q_filter_width,
                                      vars_3d_num_heads = vars_3d_num_heads)

    q <- split_heads(q, num_heads)
    k <- split_heads(k, num_heads)
    v <- split_heads(v, num_heads)

    key_depth_per_head <- key_depth %/% num_heads

    if (!vars_3d)
      q %<>% `*`(key_depth_per_head^(-0.5))

    bias <- NULL
    if (attention_type == "dot_product")
      if (max_area_width > 1 | max_area_height > 1)
        x <- dot_product_area_attention_1d(q, k, v, bias, dropout)
    else
      x <- layer_dot_product_attention_1d(q, k, v, bias, dropout)
    else
      stop("Other attention types currently unimplemented...")

    x <- combine_heads(x)

    x_shape <- shape_list2(x)

    x <-
      if (vars_3d)
        tf$get_variable("o", list(num_heads,
                                  as.integer(value_depth %/% num_heads),
                                  output_depth),
                        initializer = tf$glorot_normal_initializer) %>%
      tf$cast(x$dtype) %>%
      tf$reshape(list(value_depth, output_depth)) %>%
      {tf$tensordot(x, ., axes = 1L)}
    else
      tf$matmul(x,
                tf$get_variable(
                  name  = "output_kernel",
                  shape = list(x_shape[[length(x_shape)]], output_depth),
                  dtype = x$dtype,
                  trainable = TRUE
                ))

    x

  }, name = "multihead_attention")
}




#' Multihead attention mechanism
#' query  [batch, seqlen, depth_q]
#' memory [batch, seqlen, depth_m]
#' @export
multihead_attention <- function(query,
                                memory = NULL,
                                bias = NULL,
                                key_depth = 64L,
                                value_depth = 64L,
                                output_depth = 128L,
                                num_heads = 4L,
                                dropout = 0,
                                attention_type = "dot_product",
                                q_filter_width = 1L,
                                kv_filter_width = 1L,
                                q_padding = "VALID",
                                kv_padding = "VALID",
                                max_area_width = 1L,
                                max_area_height = 1L,
                                memory_height = 1L,
                                area_key_mode = "mean",
                                area_value_mode = "sum",
                                vars_3d = FALSE) {

  stopifnot(key_depth %% num_heads == 0, value_depth %% num_heads == 0)

  vars_3d_num_heads <- if (vars_3d) num_heads else 0

  c(q, k, v) %<-% layer_compute_qkv(query,
                                    memory,
                                    key_depth,
                                    value_depth,
                                    q_filter_width,
                                    vars_3d_num_heads = vars_3d_num_heads)

  q <- split_heads(q, num_heads)
  k <- split_heads(k, num_heads)
  v <- split_heads(v, num_heads)

  key_depth_per_head <- key_depth %/% num_heads

  if (!vars_3d)
    q %<>% `*`(key_depth_per_head^(-0.5))

  bias <- NULL
  if (attention_type == "dot_product")
    if (max_area_width > 1 | max_area_height > 1)
      x <- dot_product_area_attention_1d(q, k, v, bias, dropout)
  else
    x <- layer_dot_product_attention_1d(q, k, v, bias, dropout)
  else
    stop("Other attention types currently unimplemented...")

  x <- combine_heads(x)
  x <- layer_dense(x, output_depth, use_bias = FALSE, name = "output_transform")

  x
}



# TODO: Make this an R6 layer?
#' Input query, key, and value matrices are used to compute dot product
#' attention. (Vaswani et al. 2017)
#' q: a Tensor with shape [batch, length_q,  depth_k]
#' k: a Tensor with shape [batch, length_kv, depth_k]
#' v: a Tensor with shape [batch, length_kv, depth_v]
#' @export
dot_product_attention_1d <-
  function(q,
           k,
           v,
           bias = NULL,
           dropout = 0,
           name = "dot_product_attention") {
    q_shape <- shape_list2(q)
    scalar  <-
      tf$math$rsqrt(tf$cast(q_shape[[length(q_shape)]], tf$float32))
    logits  <- tf$matmul(q * scalar, k, transpose_b = TRUE)

    if (!is.null(bias))
      logits <- logits + bias

    weights <- tf$nn$softmax(logits, name = "attention_weights")

    x <- tf$matmul(weights, v)

    x
  }


#' Input query, key, and value matrices are used to compute dot product
#' attention. (Vaswani et al. 2017)
#' q: a Tensor with shape [batch, length_q,  depth_k]
#' k: a Tensor with shape [batch, length_kv, depth_k]
#' v: a Tensor with shape [batch, length_kv, depth_v]
#' @export
layer_dot_product_attention_1d <-
  function(q,
           k,
           v,
           bias = NULL,
           dropout = 0,
           name = "dot_product_attention") {
    layer_lambda(c(q, k, v), function(x) {
      c(q, k, v) %<-% x

      q_shape <- shape_list2(q)

      scalar <-
        tf$math$rsqrt(tf$cast(q_shape[[length(q_shape)]], tf$float32))
      logits <- tf$matmul(q * scalar, k, transpose_b = TRUE)

      if (!is.null(bias))
        logits <- logits + bias

      weights <- tf$nn$softmax(logits, name = "attention_weights")

      x <- tf$matmul(weights, v)

      x
    }, name = name)
  }


#' Simplified Self attention layer
#' Expecting shape(x) == (batch, maxtime, units)
# Ref: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L5020
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



#' antecedent: Tensor with shape [batch, length, channels]
#' depth: specifying projection layer depth
#' filter_width: how wide should the attention component be
#' padding: must be in: c("valid", "same", "left")
.compute_attention_component <- function(antecedent,
                                         depth,
                                         filter_width = 1L,
                                         padding = 'valid',
                                         name = 'c',
                                         vars_3d_num_heads = 0L) {
  layer_lambda(x, function(x) {

    if (vars_3d_num_heads > 0) {
      stopifnot(filter_width == 1)

      input_shape <- shape_list2(antecedent)
      input_depth <- input_shape[[length(input_shape)]]
      depth_per_head <- depth %/% vars_3d_num_heads
      stddev <- input_depth^(-0.5)

      if ("q" %in% name) stddev %<>% `*`(depth_per_head^(-0.5))

      var <- tf$compat$v1$get_variable(
        name,
        shape = list(
          input_depth,
          vars_3d_num_heads,
          as.integer(depth %/% vars_3d_num_heads)
        ),

        initializer = tf$random_normal_initializer(stddev = stddev))

      var %<>%
        tf$cast(dtype = antecedent$dtype) %>%
        tf$reshape(shape = list(input_depth, depth))

      return(tf$tensordot(antecedent, var, axes = 1L))
    }

    out <-
      if (filter_width == 1L)
        layer_dense(antecedent, depth, use_bias = FALSE, name = name)
    else
      layer_conv_1d(antecedent, depth, filter_width, padding = padding, name = name)

    out
  }, name = "compute_attention_component")
}



#' Split input into query, key, value matrices in preparation for
#' passing to an attention layer
#'
#' Uses .compute_attention_component function vie T2T framework
#'
#' @export
layer_compute_qkv <- function(query,
                              memory = NULL,
                              key_depth = 64L,
                              value_depth = 64L,
                              q_filter_width = 1L,
                              kv_filter_width = 1L,
                              q_padding = 'valid',
                              kv_padding = 'valid',
                              vars_3d_num_heads = 0L) {

  x <- if(!is.null(memory)) c(query, memory) else query

  layer_lambda(x, function(x) {

    if (typeof(x) == "list")
      c(query, memory) %<-% x
    else
      query <- x

    if (is.null(memory))
      memory <- query
    q <- .compute_attention_component(query,
                                      key_depth,
                                      q_filter_width,
                                      q_padding,
                                      name = "q",
                                      vars_3d_num_heads = vars_3d_num_heads)

    k <- .compute_attention_component(memory,
                                      key_depth,
                                      kv_filter_width,
                                      kv_padding,
                                      name = "k",
                                      vars_3d_num_heads = vars_3d_num_heads)

    v <- .compute_attention_component(memory,
                                      key_depth,
                                      kv_filter_width,
                                      kv_padding,
                                      name = "v",
                                      vars_3d_num_heads = vars_3d_num_heads)

    c(q, k, v)

  }, name = "layer_compute_qkv")
}



#' Split input into query, key, value matrices in preparation for
#' passing to an attention layer
#'
#' Done a little differently than T2T framework.  TODO: (TEST THIS!)
#' @export
layer_compute_qkv_v2 <-
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


#' Strided block local self-attention.
#'
#' The sequence is divided into blocks of length block_length.
#' Attention for agiven query position can see all memory positions
#' in the corresponding block and filter_width many positions to
#' the left and right of the block.
#' q Tensor [batch, heads, length, depth_k]
#' k Tensor [batch, heads, length, depth_k]
#' v Tensor [batch, heads, length, depth_v]
#' Returns Tensor [batch, heads, length, depth_v]
#' @export
local_attention_1d <-
  function(q,
           k,
           v,
           block_length = 128L,
           filter_width = 100L,
           name = NULL){
    # Shape assertions go here
    q_shape <- shape_list2(q)

    c(batch, num_heads, original_length, original_depth) %<-% q_shape

    pad_to_multiple <- function(x, pad_length) {
      x_length <- shape_list2(x)[[3]]
      tf$pad(x, list(c(0L, 0L),
                     c(0L, 0L),
                     c(0L, -x_length %% pad_length),
                     c(0L, 0L)))
    }

    pad_l_and_r <- function(x, pad_length) {
      x_length <- shape_list2(x)[[3]]
      tf$pad(x, list(c(0L, 0L),
                     c(0L, 0L),
                     c(pad_length, pad_length),
                     c(0L, 0L)))
    }

    # Set up query blocks.
    # [batch, heads, blocks_q, block_length, depth_k]
    q <- pad_to_multiple(q, block_length)
    q <- reshape_by_blocks(q, shape_list2(q), block_length)

    total_query_blocks <- shape_list2(q)[[3]]


    blocks_per_filter_width <- as.integer(filter_width %/% block_length)
    remaining <- filter_width %% block_length

    k <- pad_to_multiple(k, block_length)
    v <- pad_to_multiple(v, block_length)
    k <- pad_l_and_r(k, filter_width + block_length - remaining)
    v <- pad_l_and_r(v, filter_width + block_length - remaining)
    k <- reshape_by_blocks(k, shape_list2(k), block_length)
    v <- reshape_by_blocks(v, shape_list2(v), block_length)

    total_kv_blocks <- shape_list2(k)[[3]]

    if (remaining) {
      left_partial_block_k <- tf$slice(
        k, list(0L, 0L, 0L, block_length - remaining, 0L),
        list(-1L, -1L, total_query_blocks, -1L, -1L)
      )
      left_partial_block_v <- tf$slice(
        k, list(0L, 0L, 0L, block_length - remaining, 0L),
        list(-1L, -1L, total_query_blocks, -1L, -1L)
      )
      right_partial_block_k = tf$slice(
        k, list(0L, 0L, total_kv_blocks - total_query_blocks, 0L, 0L),
        list(-1L, -1L, -1L, remaining, -1L)
      )
      right_partial_block_v = tf$slice(
        k, list(0L, 0L, total_kv_blocks - total_query_blocks, 0L, 0L),
        list(-1L, -1L, -1L, remaining, -1L)
      )

      slices <- list(c(left_partial_block_k, left_partial_block_v),
                     c(right_partial_block_k, right_partial_block_v))
    }

    # Prepare the rest of the blocks
    first_block_index <- if (remaining) 1L else 0L
    attention_blocks  <- 2 * blocks_per_filter_width + 1L

    n <- first_block_index:attention_blocks + first_block_index

    blocks <- lapply(1:n, function(i) {
      block_k <- tf$slice(k, list(0L, 0L, i, 0L, 0L),
                          list(-1L, -1L, total_query_blocks, -1L, -1L))
      block_v <- tf$slice(k, list(0L, 0L, i, 0L, 0L),
                          list(-1L, -1L, total_query_blocks, -1L, -1L))
      c(block_k, block_v)
    })

    slices <- append(slices, blocks)

    k <- tf$concat(lapply(slices, function(b) b[[1]]), axis = 3L)
    v <- tf$concat(lapply(slices, function(b) b[[2]]), axis = 3L)

    attention_bias <- tf$expand_dims(embedding_to_padding(k) * -1e9, axis = -2L)
    shape_v <- shape_list2(v)
    depth_v <- shape_v[[length(shape_v)]]

    output <-
      dot_product_attention_1d(q, k, v, attention_bias, name = "local_1d") %>%
      tf$reshape(list(batch, num_heads, original_length, depth_v))

    # Remove the padding if introduced.
    output <- tf$slice(output,
                       list(0L, 0L, 0L, 0L),
                       list(-1L, -1L, original_length, -1L))

    output$set_shape(list(batch, num_heads, original_length, depth_v))

    output
  }



#' Strided block local self-attention.
#'
#' The sequence is divided into blocks of length block_length.
#' Attention for agiven query position can see all memory positions
#' in the corresponding block and filter_width many positions to
#' the left and right of the block.
#' q Tensor [batch, heads, length, depth_k]
#' k Tensor [batch, heads, length, depth_k]
#' v Tensor [batch, heads, length, depth_v]
#' Returns Tensor [batch, heads, length, depth_v]
#' @export
layer_local_attention_1d <- function(q,
                                     k,
                                     v,
                                     block_length = 1024L,
                                     filter_width = 100L,
                                     name = "local_attention_1d") {
  layer_lambda(x, function(x) {
    # Shape assertions go here
    q_shape <- shape_list2(q)

    c(batch, num_heads, original_length, original_depth) %<-% q_shape

    pad_to_multiple <- function(x, pad_length) {
      x_length <- shape_list2(x)[[3]]
      tf$pad(x, list(c(0L, 0L),
                     c(0L, 0L),
                     c(0L, -x_length %% pad_length),
                     c(0L, 0L)))
    }

    pad_l_and_r <- function(x, pad_length) {
      x_length <- shape_list2(x)[[3]]
      tf$pad(x, list(c(0L, 0L),
                     c(0L, 0L),
                     c(pad_length, pad_length),
                     c(0L, 0L)))
    }

    # Set up query blocks.
    # [batch, heads, blocks_q, block_length, depth_k]
    q <- pad_to_multiple(q, block_length)
    q <- reshape_by_blocks(q, shape_list2(q), block_length)

    total_query_blocks <- shape_list2(q)[[3]]


    blocks_per_filter_width <- as.integer(filter_width %/% block_length)
    remaining <- filter_width %% block_length

    k <- pad_to_multiple(k, block_length)
    v <- pad_to_multiple(v, block_length)
    k <- pad_l_and_r(k, filter_width + block_length - remaining)
    v <- pad_l_and_r(v, filter_width + block_length - remaining)
    k <- reshape_by_blocks(k, shape_list2(k), block_length)
    v <- reshape_by_blocks(v, shape_list2(v), block_length)

    total_kv_blocks <- shape_list2(k)[[3]]

    if (remaining) {
      left_partial_block_k <- tf$slice(
        k, list(0L, 0L, 0L, block_length - remaining, 0L),
        list(-1L, -1L, total_query_blocks, -1L, -1L)
      )
      left_partial_block_v <- tf$slice(
        k, list(0L, 0L, 0L, block_length - remaining, 0L),
        list(-1L, -1L, total_query_blocks, -1L, -1L)
      )
      right_partial_block_k = tf$slice(
        k, list(0L, 0L, total_kv_blocks - total_query_blocks, 0L, 0L),
        list(-1L, -1L, -1L, remaining, -1L)
      )
      right_partial_block_v = tf$slice(
        k, list(0L, 0L, total_kv_blocks - total_query_blocks, 0L, 0L),
        list(-1L, -1L, -1L, remaining, -1L)
      )

      slices <- list(c(left_partial_block_k, left_partial_block_v),
                     c(right_partial_block_k, right_partial_block_v))
    }

    # Prepare the rest of the blocks
    first_block_index <- if (remaining) 1L else 0L
    attention_blocks  <- 2 * blocks_per_filter_width + 1L

    n <- first_block_index:attention_blocks + first_block_index

    blocks <- lapply(1:n, function(i) {
      block_k <- tf$slice(k, list(0L, 0L, i, 0L, 0L),
                          list(-1L, -1L, total_query_blocks, -1L, -1L))
      block_v <- tf$slice(k, list(0L, 0L, i, 0L, 0L),
                          list(-1L, -1L, total_query_blocks, -1L, -1L))
      c(block_k, block_v)
    })

    slices <- append(slices, blocks)

    k <- tf$concat(lapply(slices, function(b) b[[1]]), axis = 3L)
    v <- tf$concat(lapply(slices, function(b) b[[2]]), axis = 3L)

    attention_bias <- tf$expand_dims(embedding_to_padding(k) * -1e9, axis = -2L)
    shape_v <- shape_list2(v)
    depth_v <- shape_v[[length(shape_v)]]

    output <-
      layer_dot_product_attention_1d(
        q, k, v, attention_bias, name = "local_1d") %>%
      tf$reshape(list(batch, num_heads, original_length, depth_v))

    # Remove the padding if introduced.
    output <- tf$slice(output,
                       list(0L, 0L, 0L, 0L),
                       list(-1L, -1L, original_length, -1L))

    output$set_shape(list(batch, num_heads, original_length, depth_v))

    output
  }, name = name)
}



# Via tensor2tensor framework
# Strided block local self-attention.
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L3118
# https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py#L1858
LocalSelfAttentionTF <- R6::R6Class(
  "LocalSelfAttentionTF",

  inherit = KerasLayer,

  public = list(
    initialize = function() {},

    build = function() {},

    call = function(x, mask = NULL) {
      # Score by attention type

      # Pass through activation to get alignments

      #
    },

    compute_output_shape = function() {}

  )
)


layer_local_self_attentionTF <-
  function(object,
           units = 32L,
           attention_width = 3L,
           attention_type = "additive",
           return_attention = FALSE,
           mask = FALSE,
           kernel_initializer = 'glorot_normal',
           bias_initializer = 'zeros') {

    create_layer(LocalSelfAttentionTF,
                 object,
                 list(units = as.integer(units),
                      attention_width = as.integer(attention_width),
                      attention_type = attention_type,
                      return_attention = return_attention,
                      mask = mask,
                      kernel_initializer = tf$keras$initializers$get(kernel_initializer),
                      bias_initializer = tf$keras$initializer$get(bias_initializer)
                 )
    )
  }





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



