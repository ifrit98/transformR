
#' Grabs list of tensor dims statically, where possible.
shape_list <- 
  function(x) {
    
    x <- tf$convert_to_tensor(x)
    
    dims <- x$get_shape()$dims
    if (is.null(dims)) return(tf$shape(x))
    
    sess <- tf$keras$backend$get_session()
    
    shape <- tf$shape(x)$eval(session = sess)
    
    ret <- vector('list', length(dims))
    
    purrr::map2(dims, shape, function(x, y) {
      dim <- x
      
      if (is.null(dim)) 
        dim <- y

      dim
      
    })
  }

#' Can we cheat and call value on Dimension 
#' class object without getting into trouble?
shape_list2 <- 
  function(x) {
    
    x <- tf$convert_to_tensor(x)
    
    dims <- x$get_shape()$dims    
    if (is.null(dims)) return(tf$shape(x))
    
    dims <- purrr::map(dims, ~.$value)
    
    sess <- tf$keras$backend$get_session()
    shape <- tf$shape(x)$eval(session = sess)
    
    ret <- vector('list', length(dims))
    
    purrr::map2(dims, shape, function(x, y) {
      dim <- x
      
      if (is.null(dim)) 
        dim <- y
      
      dim
      
    })
  }
