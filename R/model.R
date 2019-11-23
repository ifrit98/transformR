# ENCODER STEP
# Self attention
# feed-forward layer

# x <- tf$random$normal(16L, 256L, 128L)
# y <- multihead_attention( layer_preprocess( x ) )
# x <- layer_postprocess( y )
# y <- feed_forward_layer( layer_preprocess( x ) )
# x <- layer_postprocess( y )
# return( layer_preprocess( x ) )

#' Applies the specified normalization type to input x
apply_norm <-
  function(x, norm_type, depth = tail(shape_list2(x), 1), eps = 1e-9) {
    if (!norm_type %in% c("layer", "group", "batch", "noam", "l2"))
      stop("norm_type must be one of: 'layer', 'batch', 'noam', 'lr'.")
    switch(
      norm_type,
      "layer" = x, # layer_normalization(x)  TODO: Implement!
      "batch" = layer_batch_normalization(x, epsilon = eps),
      "group" = x, # group_normalization(x), TODO: implement!
      "noam"  = x, # noam_normalization(x),  TODO: implement!
      "l2"    = k_l2_normalize(x, -1L)
    )
  }

#' Apply a sequence of functions to the input or output of a layer.
#'
#'The sequence is specified as a string which may contain the following
#'characters:
#'  a: add previous_value
#'  n: apply normalization
#'  d: apply dropout
#'  z: zero add
#'
#' For example, if sequence=="dna", then the output is
#'   previous_value + normalize(dropout(x))
layer_prepost_process <-
  function(resid,
           x,
           sequence = NULL,
           drpoout = 0,
           norm_type = NULL,
           depth = tail(shape_list2(x), 1),
           eps = 1e-9,
           name = NULL) {

  if (is.null(sequence)) return(x)

  for (c in sequence) {
    x <- switch(
      c,
      a = layer_add(x, resid),
      z = zero_add(resid, x),
      n = apply_norm(x, norm_type, depth, eps)
    )
  }

  x
}


layer_preprocess <- function(x) {

}

layer_postprocess <- function(x) {

}

#' Hidden layer with RELU activation followed by linear projection.
dense_relu_dense <- function(x,
                             filters,
                             output_depth,
                             first_kernel_size = 3L,
                             second_kernel_size = 3L,
                             output_activation = NULL,
                             padding = "SAME",
                             dropout = 0) {
    h <- layer_dense(x,
                     filters,
                     use_bias = TRUE,
                     activation = 'relu',
                     dropout = dropout,
                     name = "dense1")

    o <- layer_dense(x,
                     output_depth,
                     use_bias = TRUE,
                     activation = output_depth,
                     name = "dense2")
    o
}

#' Hidden layer with RELU activation followed by linear projection.
conv_relu_conv <- function(x,
                           filters,
                           output_depth,
                           first_kernel_size = 3L,
                           second_kernel_size = 3L,
                           padding = "SAME",
                           dropout = 0) {

  h <- layer_conv_1d(x,
                     filters,
                     first_kernel_size,
                     activation = "relu",
                     dropout = dropout,
                     padding = padding,
                     name = "conv1")

  return(layer_conv_1d(h,
                       output_depth,
                       second_kernel_size,
                       padding = padding,
                       name = "conv2"))

}

#' Hidden layer with RELU activation followed by linear projection
sepconv_relu_sepconv <- function(x,
                                 filters,
                                 output_depth,
                                 first_kernel_size = list(1L, 1L),
                                 second_kernel_size = list(1L, 1L),
                                 padding = "LEFT",
                                 dropout = 0) {
  h <-
    layer_separable_conv_1d(
      x,
      filters,
      first_kernel_size,
      padding = padding,
      dropout = dropout,
      activation = 'relu',
      name = "sepconv1"
    )

  o <-
    layer_separable_conv_1d(h,
                            output_depth,
                            second_kernel_size,
                            padding = padding,
                            name = "sepconv2")
}



feed_forward_layer <- function(x,
                               layer_type = "dense_relu_dense",
                               conv_pad = "LEFT",
                               pad_remover = NULL) {
  stopifnot(layer_type %in% c("dense_relu_dense",
                              "conv_relu_conv",
                              "sepconv_relu_sepconv"))
  filters <- 32L
  hidden  <- 64L
  dropout <- 0

  if (layer_type == "dense_relu_dense") {
    if (!is.null(pad_remover)) {
      og_shape <- shape_list2(x)

      # Collapse x along examples and remove padding for speedup
      x <- tf$reshape(x, tf$concat(list(c(-1L), og_shape[3:length(og_shape)]), axis = 0L))
      x <- tf$expand_dims(pad_remover$remove(x), axis = 0L)
    }

    output <- dense_relu_dense(x, filters, hidden, dropout)

    if (!is.null(pad_remover))
      output <- tf.reshape(
        pad_remover$restore(tf$squeeze(output, axis = 0L)), og_shape)

    return(out)
  }

  if (layer_type == "conv_relu_conv")
    return(conv_relu_conv(x,
                          filters = filters,
                          hidden = hidden,
                          first_kernel_size = first_kernel_size,
                          second_kernel_size = 1L,
                          padding = conv_pad,
                          dropout = dropout,
                          decode_loop_step = decode_loop_step))

  if (layer_type == "sepconv_relu_sepconv")
    return(sepconv_relu_sepconv(x,
                                filters = filters,
                                hidden = hidden,
                                first_kernel_size = list(3L, 1L),
                                second_kernel_size = list(31L,1L),
                                padding = "LEFT",
                                dropout = dropout))

}
