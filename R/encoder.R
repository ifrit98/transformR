

default_transformer_hparams <- function() {
  list(
    embedding_filter       = 32L,
    embedding_kernel       = 3L,
    embedding_stride       = 1L,
    embedding_padding      = "SAME",
    preprocess_seq         = "n",
    attention_key_depth    = 64L,
    attention_val_depth    = 64L,
    attention_out_depth    = 128L,
    attention_num_heads    = 4L,
    attention_dropout      = 0,
    attention_type         = "dot_product",
    q_filter_width         = 3L,
    kv_filter_width        = 3L,
    attention_vars3d       = FALSE,
    postprocess_sequence   = "dna",
    feed_forward_depth     = 64L,
    final_output_depth     = 128L,
    feed_forward_dropout   = 0,
    final_process_sequence = "dna"
  )
}


#' Define Transformer encoder function
#' @export
transformer_encoder_v2 <- function(x,
                                   embedding_function = NULL,
                                   hparams = default_transformer_hparams()) {

  if (is.null(embedding_function))
    x <-
      layer_conv_1d(x,
                    hparams$embedding_filter,
                    hparams$embedding_kernel,
                    hparams$embedding_stride,
                    hparams$embedding_padding,
                    activation = 'relu')

  processed_input <-
    layer_preprocess(x, hparams$preprocess_seq, name = "preprocess_in")

  attention_out <-
    multihead_attention(
      processed_input,
      key_depth       = hparams$attention_key_depth,
      value_depth     = hparams$attention_val_depth,
      output_depth    = hparams$attention_out_depth,
      num_heads       = hparams$attention_num_heads,
      dropout         = hparams$attention_dropout,
      attention_type  = hparams$attention_type,
      q_filter_width  = hparams$q_filter_width,
      kv_filter_width = hparams$kv_filter_width,
      vars_3d         = hparams$attention_vars3d
    )

  processed_output <-
    layer_postprocess(x,
                      attention_out,
                      hparams$postprocess_sequence,
                      name = "postprocess_atn")

  feed_forward_out <-
    feed_forward_layer(
      processed_output,
      filters = hparams$feed_forward_depth,
      hidden_depth = hparams$final_output_depth,
      dropout = hparams$feed_forward_dropout
    )

  encoder_output <-
    layer_postprocess(processed_output,
                      feed_forward_out,
                      sequence = hparams$final_process_sequence,
                      name = "postprocess_ff")

  encoder_output
}



#' Define Transformer encoder function with lambda layer components
transformer_encoder_v3 <- function(x,
                                   embedding_function = NULL,
                                   hparams = default_transformer_hparams()) {

  if (is.null(embedding_function))
    x <-
      layer_conv_1d(x,
                    hparams$embedding_filter,
                    hparams$embedding_kernel,
                    hparams$embedding_stride,
                    hparams$embedding_padding,
                    activation = 'relu')

  processed_input <-
    layer_preprocess(x, hparams$preprocess_seq, name = "preprocess_in")

  attention_out <-
    layer_multihead_attention(
      processed_input,
      key_depth       = hparams$attention_key_depth,
      value_depth     = hparams$attention_val_depth,
      output_depth    = hparams$attention_out_depth,
      num_heads       = hparams$attention_num_heads,
      dropout         = hparams$attention_dropout,
      attention_type  = hparams$attention_type,
      q_filter_width  = hparams$q_filter_width,
      kv_filter_width = hparams$kv_filter_width,
      vars_3d         = hparams$attention_vars3d
    )

  processed_output <-
    layer_postprocess(x,
                      attention_out,
                      hparams$postprocess_sequence,
                      name = "postprocess_atn")

  feed_forward_out <-
    layer_feed_forward(
      processed_output,
      filters = hparams$feed_forward_depth,
      hidden_depth = hparams$final_output_depth,
      dropout = hparams$feed_forward_dropout
    )

  encoder_output <-
    layer_postprocess(processed_output,
                      feed_forward_out,
                      sequence = hparams$final_process_sequence,
                      name = "postprocess_ff")

  encoder_output
}
