% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/attention-utils.R
\name{create_qkv}
\alias{create_qkv}
\title{Takes input tensor of shape [batch, seqlen, channels] and
creates query, key, and value tensors to pass to attention
mechanisms downstream.}
\usage{
create_qkv(x, filter_depth, num_parts = 1L, share_kv = FALSE)
}
\description{
query shape [batch, seqlen, filter_depth]
key shape   [batch, seqlen, filter_depth]
value shape [batch, seqlen, filter_depth]
}
