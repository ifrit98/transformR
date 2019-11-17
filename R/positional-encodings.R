
#' Gets a timing signal for a given length and number of channels
#'
#' Gets a bunch of sinusoids of different frequencies.
#' Each channel of the input Tensor is incremented by a sinusoid of a different
#' frequency and phase.
#' This allows attention to learn to use absolute and relative positions.
#' Timing signals should be added to some precursors of both the query and the
#' memory inputs to attention.
#' The use of relative position is possible because sin(x+y) and cos(x+y) can
#' be expressed in terms of y, sin(x) and cos(x).
#' In particular, we use a geometric sequence of timescales starting with
#' min_timescale and ending with max_timescale.  The number of different
#' timescales is equal to channels / 2. For each timescale, we
#' generate the two sinusoidal signals sin(timestep/timescale) and
#' cos(timestep/timescale).  All of these sinusoids are concatenated in
#' the channels dimension.
#'
#' @param length scalar, length of timing signal sequence.
#' @param channels scalar, size of timing embeddings to create.
#' The number of different timescales is equal to channels / 2.
#' @param min_timescale: a float
#' @param max_timescale: a float
#' @param start_index: index of first position
#' @exoport
get_timing_signal_1d <-
  function(length,
           channels,
           min_timescale = 1,
           max_timescale = 1e4,
           start_index = 0) {

    position <- tf$to_float(tf$range(length) + start_index)

    num_timescales <- channels %/% 2L

    log_timescale_increment <-
      (log(max_timescale) / min_timescale) / (tf$to_float(num_timescales) - 1)

    inv_timescales <- min_timescale * tf$exp(
      tf$to_float(tf$range(num_timescales)) * -log_timescale_increment)

    scaled_time <-
      tf$expand_dims(position, 1L) * tf$expand_dims(inv_timescales, 0L)

    signal <-
      tf$concat(list(tf$sin(scaled_time), tf$cos(scaled_time)), axis = 1L)

    signal <-
      tf$pad(signal, list(c(0L, 0L), c(0L, tf$mod(channels, 2L))))

    signal <-
      tf$reshape(signal, list(1L, as.integer(length), as.integer(channels)))

    signal

  }



#' Add timing signal to a tensor.
#'
#' Adds a bunch of sinusoids of different frequencies to a Tensor.
#' Each channel of the input Tensor is incremented by a sinusoid of a different
#' frequency and phase.
#' This allows attention to learn to use absolute and relative positions.
#' Timing signals should be added to some precursors of both the query and the
#' memory inputs to attention.
#' The use of relative position is possible because sin(x+y) and cos(x+y) can
#' be experessed in terms of y, sin(x) and cos(x).
#' In particular, we use a geometric sequence of timescales starting with
#' min_timescale and ending with max_timescale.  The number of different
#' timescales is equal to channels / 2. For each timescale, we
#' generate the two sinusoidal signals sin(timestep/timescale) and
#' cos(timestep/timescale).  All of these sinusoids are concatenated in
#' the channels dimension.
#'
#' @param x: a Tensor with shape [batch, length, channels]
#' @param min_timescale: a float
#' @param max_timescale: a float
#' @param start_index: index of first position
#' @export
add_timing_signal_1d <-
  function(x,
           min_timescale = 1,
           max_timescale = 1e4,
           start_index = 0L) {

    shape    <- shape_list2(x)

    length   <- shape[[2]]
    channels <- shape[[3]]

    signal <-
      get_timing_signal_1d(length, channels, min_timescale,
                           max_timescale, start_index)

    x + signal
  }
