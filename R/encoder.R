

default_transformer_hparams <- function() {
  list(
    embedding_filter       = 32L,
    embedding_kernel       = 3L,
    preprocess_seq         = "n",
    attention_key_depth    = 64L,
    attention_val_depth    = 64L,
    attention_out_depth    = 128L,
    attention_num_heads    = 4L,
    attention_dropout      = 0,
    attention_type         = "dot_product",
    q_filter_width         = 5L,
    kv_filter_width        = 5L,
    attention_vars3d       = FALSE,
    postprocess_sequence   = "dna",
    feed_forward_depth     = 128L,
    hidden_depth           = 64L,
    feed_forward_dropout   = 0,
    final_process_sequence = "dna"
  )
}


#' Define Transformer encoder function
#' TODO: Why does default hparams result in shape mismatch for compute_qkv?
transformer_encoder <- function(x,
                                embedding_function = NULL,
                                hparams = default_transformer_hparams()) {
  # Make embeddings or extract features...
  if (is.null(embedding_function))
    x <-
      layer_conv_1d(x,
                    hparams$embedding_filter,
                    hparams$embedding_kernel,
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
      hidden_depth = hparams$hidden_depth,
      dropout = hparams$feed_forward_dropout
    )

  encoder_output <-
    layer_postprocess(processed_output,
                      feed_forward_out,
                      sequence = hparams$final_process_sequence,
                      name = "postprocess_ff")

  encoder_output
}
