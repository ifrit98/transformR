

#' Define Transformer encoder function
transformer_encoder <- function(x) {

  # Make embeddings or extract features...
  x <- layer_conv_1d(x, filters = 32L, kernel_size = 3L, activation = 'relu')

  processed_input <- layer_preprocess(x, "n", name = "preprocess_in")

  attention_out <- multihead_attention(processed_input)

  processed_output <-
    layer_postprocess(x, attention_out, "dna", name = "postprocess_atn")

  feed_forward_out <-
    feed_forward_layer(
      processed_output,
      filters = 32L,
      hidden_depth = 64L,
      dropout = 0.3
    )

  encoder_output   <-
    layer_postprocess(processed_output,
                      feed_forward_out,
                      sequence = "dna",
                      name = "postprocess_ff")


  encoder_output
}


#' Add residual connection and project if feature dimensions do not match
#' @export
add_residual <- function(x, resid) {
  x_depth <- tail(shape_list2(x), 1)[[1]]
  r_depth <- tail(shape_list2(resid), 1)[[1]]

  if(x_depth != r_depth)
    resid <- layer_conv_1d(resid, x_depth, 1L)

  layer_add(list(x, resid))
}


#' Adds residual information to current output
#' @export
layer_add_residual <- function(x, resid) {
  inputs <- c(x, resid)
  layer_lambda(inputs, function(inputs) {
    x     <- inputs[[1]]
    resid <- inputs[[2]]

    x_depth <- tail(shape_list2(x), 1)[[1]]
    r_depth <- tail(shape_list2(resid), 1)[[1]]

    if(x_depth != r_depth)
      resid <- layer_conv_1d(resid, x_depth, 1L)

    layer_add(list(x, resid))

  })
}


#' Applies specified normalization type to input x
#' @export
apply_normalization <-
  function(x,
           norm_type,
           depth = tail(shape_list2(x), 1),
           eps = 1e-9) {
    if (!norm_type %in% c("layer", "batch", "l2"))
      stop("norm_type must be one of: 'layer', 'batch', 'l2'.")
    switch(
      norm_type,
      "layer" = layer_normalization(x),
      "batch" = layer_batch_normalization(x, epsilon = eps),
      "group" = x, # TODO: implement
      "noam"  = x, # TODO: Implement
      "l2"    = k_l2_normalize(x,-1L) # TODO: make R6 wrapper for tracking?
    )
  }


#' Apply normalization function to input tensor
#' @export
layer_apply_normalization <-
  function(x,
           norm_type,
           depth = tail(shape_list2(x), 1),
           eps = 1e-9) {
    layer_lambda(x, function(x) {
      if (!norm_type %in% c("layer", "batch", "l2"))
        stop("norm_type must be one of: 'layer', 'batch', 'l2'.")
      switch(
        norm_type,
        "layer" = layer_normalization(x),
        "batch" = layer_batch_normalization(x, epsilon = eps),
        "group" = x, # TODO: implement
        "noam"  = x, # TODO: Implement
        "l2"    = k_l2_normalize(x,-1L) # TODO: make R6 wrapper for tracking?
      )
    }, name = "apply_norm")

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
#' @export
layer_prepost_process <-
  function(resid,
           x,
           sequence = NULL,
           dropout = 0,
           norm_type = "layer",
           depth = tail(shape_list2(x), 1),
           eps = 1e-9,
           name = NULL) {

  if (is.null(sequence)) return(x)

  for (c in strsplit(sequence, "") %>% unlist) {

    stopifnot(c %in% c("a", "d", "n"))

    if (c == "a" & is.null(resid))
      stop("Sequence must not contain 'a' to add without a residual connection")

    x <- switch(
      c,
      a = add_residual(x, resid),
      d = layer_dropout(x, dropout),
      n = apply_normalization(x, norm_type, depth, eps)
    )
  }

  x
}


#' @export
layer_prepost_process_v2 <-
  function(resid,
           x,
           sequence = NULL,
           dropout = 0,
           norm_type = "layer",
           depth = tail(shape_list2(x), 1),
           eps = 1e-9,
           name = NULL) {

    if (is.null(sequence)) return(x)

    inputs <- c(resid, x)

    layer_lambda(inputs, function(inputs) {
      resid <- inputs[[1]]
      x     <- inputs[[2]]

      for (c in strsplit(sequence, "") %>% unlist) {

        stopifnot(c %in% c("a", "d", "n"))

        if (c == "a" & is.null(resid))
          stop("Sequence must not contain 'a' without a residual connection")

        x <- switch(
          c,
          a = add_residual(x, resid),
          d = layer_dropout(x, dropout),
          n = apply_normalization(x, norm_type, depth, eps)
        )
      }

      x


    }, name = name)
  }


#' Preprocess layer input by applying a sequence of functions
#' @export
layer_preprocess <-
  function(layer_input,
           sequence = NULL,
           dropout = 0,
           norm_type = "layer",
           name = NULL,
           v2 = FALSE) {
    if (!v2)
      return(layer_prepost_process(
        resid     = NULL,
        x         = layer_input,
        sequence  = sequence,
        dropout   = dropout,
        norm_type = norm_type,
        name      = name
      ))
    else
      return(layer_prepost_process_v2(
        resid     = NULL,
        x         = layer_input,
        sequence  = sequence,
        dropout   = dropout,
        norm_type = norm_type,
        name      = name
      ))
  }


#' Postprocess layer output by applying a sequence of functions
#' @export
layer_postprocess <- function(layer_input,
                              layer_output,
                              sequence = NULL,
                              dropout = 0,
                              norm_type = "layer",
                              name = NULL,
                              v2 = FALSE) {
  if (!v2)
    return(layer_prepost_process(
      resid     = layer_input,
      x         = layer_output,
      sequence  = sequence,
      dropout   = dropout,
      norm_type = norm_type,
      name      = name
    ))
  else
    return(layer_prepost_process_v2(
      resid     = layer_input,
      x         = layer_output,
      sequence  = sequence,
      dropout   = dropout,
      norm_type = norm_type,
      name      = name
    ))

}


#' Hidden layer with RELU activation followed by linear projection.
dense_relu_dense <- function(x,
                             filters,
                             output_depth,
                             first_kernel_size = 3L,
                             second_kernel_size = 3L,
                             output_activation = NULL,
                             dropout = 0) {
  # browser()
  h <- layer_dense(x,
                   filters,
                   use_bias = TRUE,
                   activation = 'relu',
                   name = "dense1")

  if (dropout > 0) h <- layer_dropout(h, dropout)

  o <- layer_dense(x,
                   output_depth,
                   use_bias = TRUE,
                   activation = output_activation,
                   name = "dense2")
  o
}



#' Hidden layer with RELU activation followed by linear projection.
layer_dense_relu_dense <- function(x,
                             filters,
                             output_depth,
                             first_kernel_size = 3L,
                             second_kernel_size = 3L,
                             output_activation = NULL,
                             dropout = 0) {
  layer_lambda(x, function(x) {
    h <- layer_dense(x,
                     filters,
                     use_bias = TRUE,
                     activation = 'relu',
                     name = "dense1")

    if (dropout > 0) h <- layer_dropout(h, dropout)

    o <- layer_dense(x,
                     output_depth,
                     use_bias = TRUE,
                     activation = output_activation,
                     name = "dense2")
    o
  }, name = "dense_relu_dense")
}


#' Hidden layer with RELU activation followed by linear projection.
conv_relu_conv <- function(x,
                           filters,
                           output_depth,
                           first_kernel_size = 3L,
                           second_kernel_size = 3L,
                           padding = "SAME",
                           dropout = 0) {
  layer_lambda(x, function(x) {

  h <- layer_conv_1d(x,
                     filters,
                     first_kernel_size,
                     activation = "relu",
                     padding = padding,
                     name = "conv1")

  if (dropout > 0) h <- layer_dropout(h, dropout)

  return(layer_conv_1d(h,
                       output_depth,
                       second_kernel_size,
                       padding = padding,
                       name = "conv2"))

  }, name = "conv_relu_conv")

}

#' Hidden layer with RELU activation followed by linear projection
sepconv_relu_sepconv <- function(x,
                                 filters,
                                 output_depth,
                                 first_kernel_size = list(1L, 1L),
                                 second_kernel_size = list(1L, 1L),
                                 padding = "LEFT",
                                 dropout = 0) {
  layer_lambda(x, function(x) {
    h <-
      layer_separable_conv_1d(
        x,
        filters,
        first_kernel_size,
        padding = padding,
        activation = 'relu',
        name = "sepconv1"
      )

    if (dropout > 0) h <- layer_dropout(h, dropout)

    o <-
      layer_separable_conv_1d(h,
                              output_depth,
                              second_kernel_size,
                              padding = padding,
                              name = "sepconv2")

    o
  }, name = "sepconv_relu_sepconv")
}



feed_forward_layer <- function(x,
                               filters,
                               hidden_depth,
                               dropout = 0,
                               layer_type = "dense_relu_dense",
                               conv_pad = "LEFT",
                               pad_remover = NULL) {
  stopifnot(layer_type %in% c("dense_relu_dense",
                              "conv_relu_conv",
                              "sepconv_relu_sepconv"))

  if (layer_type == "dense_relu_dense") {
    if (!is.null(pad_remover)) {
      og_shape <- shape_list2(x)

      # Collapse x along examples and remove padding for speedup
      x <- tf$reshape(x, tf$concat(list(c(-1L), og_shape[3:length(og_shape)]),
                                   axis = 0L))
      x <- tf$expand_dims(pad_remover$remove(x), axis = 0L)
    }

    output <- dense_relu_dense(x, filters, hidden_depth, dropout)

    if (!is.null(pad_remover))
      output <- tf$reshape(
        pad_remover$restore(tf$squeeze(output, axis = 0L)), og_shape)

    return(output)
  }

  if (layer_type == "conv_relu_conv")
    return(conv_relu_conv(x,
                          filters = filters,
                          output_depth = hidden_depth,
                          first_kernel_size = first_kernel_size,
                          second_kernel_size = 1L,
                          padding = conv_pad,
                          dropout = dropout,
                          decode_loop_step = decode_loop_step))

  if (layer_type == "sepconv_relu_sepconv")
    return(sepconv_relu_sepconv(x,
                                filters = filters,
                                output_depth = hidden_depth,
                                first_kernel_size = list(3L, 1L),
                                second_kernel_size = list(31L,1L),
                                padding = "LEFT",
                                dropout = dropout))

}


#' Feed forward layer for transformer encoder
#' @export
layer_feed_forward <- function(x,
                               filters,
                               hidden_depth,
                               dropout = 0,
                               layer_type = "dense_relu_dense",
                               conv_pad = "LEFT",
                               pad_remover = NULL) {
  stopifnot(
    layer_type %in% c("dense_relu_dense", "conv_relu_conv", "sepconv_relu_sepconv"))

  layer_lambda(x, function(x) {

    if (layer_type == "dense_relu_dense") {
      if (!is.null(pad_remover)) {
        og_shape <- shape_list2(x)

        # Collapse x along examples and remove padding for speedup
        x <- tf$reshape(x, tf$concat(list(c(-1L), og_shape[3:length(og_shape)]),
                                     axis = 0L))
        x <- tf$expand_dims(pad_remover$remove(x), axis = 0L)
      }

      output <- dense_relu_dense(x, filters, hidden_depth, dropout)

      if (!is.null(pad_remover))
        output <- tf$reshape(
          pad_remover$restore(tf$squeeze(output, axis = 0L)), og_shape)

      return(output)
    }

    if (layer_type == "conv_relu_conv")
      return(conv_relu_conv(x,
                            filters = filters,
                            output_depth = hidden_depth,
                            first_kernel_size = first_kernel_size,
                            second_kernel_size = 1L,
                            padding = conv_pad,
                            dropout = dropout,
                            decode_loop_step = decode_loop_step))

    if (layer_type == "sepconv_relu_sepconv")
      return(sepconv_relu_sepconv(x,
                                  filters = filters,
                                  output_depth = hidden_depth,
                                  first_kernel_size = list(3L, 1L),
                                  second_kernel_size = list(31L,1L),
                                  padding = "LEFT",
                                  dropout = dropout))

  }, name = "layer_feed_forward")
}
