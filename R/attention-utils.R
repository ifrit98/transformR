
#' Calculates the padding mask based on which embeddings are all zero.
#' 
#' emb Tensor with shape [..., depth]
#' 
#' Returns:
#'   a float Tensor with shape [...]. Each element is 1 if its 
#'   corresponding embedding vector is all zero, and is 0 otherwise.
embedding_to_padding <- function(emb) {
  emb_sum <- tf$reduce_sum(tf$abs(emb), axis = -1L)
  tf$to_float(tf$equal(emb_sum, 0))
}


#' Reshape input by splitting length over blocks of memory_block_size.
#'
#' x Tensor [batch, heads, length, depth]
#' x_shape tf$TensorShape of x
#' memory_block_size Integer to dividing length by
#' Return 
#'   Tensor [batch, heads, length %/% memory_block_size, memory_block_size, depth]
reshape_by_blocks <- function(x, x_shape, memory_block_size) {
  x <- tf$reshape(x,
                  list(x_shape[[1]], x_shape[[2]], 
                       as.integer(x_shape[[3]] %/% memory_block_size), 
                       memory_block_size, x_shape[[4]]))
  x
}



#' Reshape x so that the last dimension becomes two dimensions.
split_last_dimension <- function(x, n) {
  x_shape <- shape_list2(x)
  
  n <- as.integer(n)
  m <- x_shape[[length(x_shape)]]
  
  stopifnot(m %% n == 0)
  
  out <- 
    tf$reshape(x, c(x_shape[-length(x_shape)], list(n, as.integer(m %/% n))))
  
  out
}



#' Split channels (dimension 2) into multiple heads (becomes dimension 1).
#' x Tensor shape: [batch, length, channels]
#' num_heads integer
split_heads <- function(x, num_heads) {
  out <- tf$transpose(split_last_dimension(x, num_heads), 
                      perm = list(0L, 2L, 1L, 3L))
  out
}



#' Reshape x so that the last two dimension become one.
combine_last_two_dimensions <- function(x) {
  x_shape <- shape_list2(x)
  c(a, b) %<-% x_shape[-c(1:(length(x_shape)-2))]
  
  tf$reshape(x, c(x_shape[c(1,2)], as.integer(a * b)))
}



#' Inverse of split_heads.
combine_heads <- function(x) {
  combine_last_two_dimensions(tf$transpose(x, list(0L, 2L, 1L, 3L)))  
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
create_qkv <- function(x, filter_depth, num_parts = 1L, share_kv = FALSE) {
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
}



#' query  [batch, length_q, channels]
#' memory [batch, length_m, channels] (optional, usually RNN hidden states)
#' return [batch, length, depth] (q, k ,v) tensors
compute_qkv <-
  function(query,
           memory = NULL,
           key_depth = 64L,
           value_depth = 64L,
           q_filter_width = 1L,
           kv_filter_width = 1L,
           q_padding = 'valid',
           kv_padding = 'valid',
           vars_3d_num_heads = 0L) {
    
    if (is.null(memory))
      memory <- query
    q <- compute_attention_component(query,
                                     key_depth,
                                     q_filter_width,
                                     q_padding,
                                     "q",
                                     vars_3d_num_heads)
    
    k <- compute_attention_component(memory,
                                     key_depth,
                                     kv_filter_width,
                                     kv_padding,
                                     "k",
                                     vars_3d_num_heads)
    
    v <- compute_attention_component(memory,
                                     key_depth,
                                     kv_filter_width,
                                     kv_padding,
                                     "v",
                                     vars_3d_num_heads)
    
    c(q, k, v)
  }



#' antecedent: Tensor with shape [batch, length, channels]
#' depth: specifying projection layer depth
#' filter_width: how wide should the attention component be
#' padding: must be in: c("valid", "same", "left")
compute_attention_component <- function(antecedent,
                                        depth,
                                        filter_width = 1L,
                                        padding = 'valid',
                                        name = 'c',
                                        vars_3d_num_heads = 0L) {
  if (vars_3d_num_heads > 0) {
    stopifnot(filter_width == 1)
    
    input_shape <- shape_list2(antecedent)
    input_depth <- input_shape[[length(input_shape)]]
    depth_per_head <- depth %/% vars_3d_num_heads
    stddev <- input_depth ^ (-0.5)
    
    if ("q" %in% name)
      stddev %<>% `*`(depth_per_head ^ (-0.5))
    
    var <- tf$compat$v1$get_variable(
      name = name,
      shape = list(
        input_depth,
        vars_3d_num_heads,
        as.integer(depth %/% vars_3d_num_heads)
      ),
      initializer = tf$random_normal_initializer(stddev = stddev),
      dtype = antecedent$dtype
    )
    
    var <- var %>% tf$reshape(shape = list(input_depth, depth))
    
    return(tf$tensordot(antecedent, var, axes = 1L))
  }
  
  out <- 
    if (filter_width == 1L) 
      layer_dense(antecedent, depth, use_bias = FALSE, name = name)
  else
    layer_conv_1d(antecedent, depth, filter_width, padding = padding, name = name)
  
  out
}

