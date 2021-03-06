% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/attention-layers.R
\name{multihead_attention}
\alias{multihead_attention}
\title{Multihead attention mechanism
query  [batch, seqlen, depth_q]
memory [batch, seqlen, depth_m]}
\usage{
multihead_attention(
  query,
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
  q_padding = "SAME",
  kv_padding = "SAME",
  max_area_width = 1L,
  max_area_height = 1L,
  memory_height = 1L,
  area_key_mode = "mean",
  area_value_mode = "sum",
  vars_3d = FALSE
)
}
\description{
Multihead attention mechanism
query  [batch, seqlen, depth_q]
memory [batch, seqlen, depth_m]
}
