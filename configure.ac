AC_INIT([simple_gpu], [1.0], [https://github.com/scemama/simple_gpu])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_AUX_DIR([build-config])

AM_INIT_AUTOMAKE([foreign -Wall -Werror color-tests parallel-tests silent-rules])
AC_PROG_CC
AC_PROG_FC
AM_PROG_AR
LT_INIT

# Check for required programs
AC_CHECK_PROG([HAVE_FC], [gfortran], [yes], [no])
if test "x$HAVE_FC" = "xno"; then
  AC_CHECK_PROG([HAVE_FC], [ifort], [yes], [no])
fi

if test "x$HAVE_FC" = "xno"; then
  AC_MSG_ERROR([No Fortran compiler found. Please install gfortran or ifort.])
fi

# Check for BLAS library (required for CPU version)
AC_ARG_WITH([blas],
  [AS_HELP_STRING([--with-blas=LIB], [use BLAS library LIB])],
  [BLAS_LIBS="$withval"],
  [AC_CHECK_LIB([openblas], [dgemm_], [BLAS_LIBS="-lopenblas"],
    [AC_CHECK_LIB([blas], [dgemm_], [BLAS_LIBS="-lblas"],
      [AC_MSG_ERROR([BLAS library not found. Please install OpenBLAS or specify with --with-blas])])])])
AC_SUBST([BLAS_LIBS])

# Option to disable NVIDIA library
AC_ARG_ENABLE([nvidia],
  [AS_HELP_STRING([--disable-nvidia], [disable NVIDIA GPU library])],
  [enable_nvidia=$enableval],
  [enable_nvidia=yes])

# Check for CUDA/cuBLAS libraries if NVIDIA is enabled
BUILD_NVIDIA=no
if test "x$enable_nvidia" = "xyes"; then
  # Check for nvcc compiler
  AC_CHECK_PROG([NVCC], [nvcc], [nvcc])
  
  if test "x$NVCC" != "x"; then
    # Try to find CUDA installation
    AC_ARG_WITH([cuda],
      [AS_HELP_STRING([--with-cuda=DIR], [CUDA installation directory])],
      [CUDA_PATH="$withval"],
      [CUDA_PATH="/usr/local/cuda"])
    
    # Check if CUDA path exists
    if test -d "$CUDA_PATH"; then
      CUDA_CFLAGS="-I$CUDA_PATH/include"
      CUDA_LDFLAGS="-L$CUDA_PATH/lib64 -L$CUDA_PATH/lib"
      
      # Save current flags
      OLD_CFLAGS="$CFLAGS"
      OLD_LDFLAGS="$LDFLAGS"
      CFLAGS="$CFLAGS $CUDA_CFLAGS"
      LDFLAGS="$LDFLAGS $CUDA_LDFLAGS"
      
      # Check for CUDA runtime library
      AC_CHECK_LIB([cudart], [cudaMalloc], [HAVE_CUDART=yes], [HAVE_CUDART=no])
      
      # Check for cuBLAS library
      AC_CHECK_LIB([cublas], [cublasDgemm], [HAVE_CUBLAS=yes], [HAVE_CUBLAS=no], [-lcudart])
      
      # Restore flags
      CFLAGS="$OLD_CFLAGS"
      LDFLAGS="$OLD_LDFLAGS"
      
      if test "x$HAVE_CUDART" = "xyes" -a "x$HAVE_CUBLAS" = "xyes"; then
        BUILD_NVIDIA=yes
        CUDA_LIBS="-lcudart -lcublas"
        AC_SUBST([CUDA_CFLAGS])
        AC_SUBST([CUDA_LDFLAGS])
        AC_SUBST([CUDA_LIBS])
        AC_SUBST([CUDA_PATH])
      else
        AC_MSG_WARN([CUDA libraries not found. NVIDIA GPU library will not be built.])
      fi
    else
      AC_MSG_WARN([CUDA installation not found at $CUDA_PATH. NVIDIA GPU library will not be built.])
    fi
  else
    AC_MSG_WARN([nvcc compiler not found. NVIDIA GPU library will not be built.])
  fi
fi

AM_CONDITIONAL([BUILD_NVIDIA], [test "x$BUILD_NVIDIA" = "xyes"])

# Output files
AC_CONFIG_FILES([
  Makefile
])

AC_OUTPUT

# Print configuration summary
echo ""
echo "========================================"
echo " simple_gpu configuration summary"
echo "========================================"
echo "  CC:              $CC"
echo "  FC:              $FC"
echo "  CFLAGS:          $CFLAGS"
echo "  FCFLAGS:         $FCFLAGS"
echo "  BLAS_LIBS:       $BLAS_LIBS"
echo ""
if test "x$BUILD_NVIDIA" = "xyes"; then
  echo "  NVIDIA GPU:      enabled"
  echo "  NVCC:            $NVCC"
  echo "  CUDA_PATH:       $CUDA_PATH"
  echo "  CUDA_CFLAGS:     $CUDA_CFLAGS"
  echo "  CUDA_LDFLAGS:    $CUDA_LDFLAGS"
  echo "  CUDA_LIBS:       $CUDA_LIBS"
else
  echo "  NVIDIA GPU:      disabled"
fi
echo "========================================"
echo ""
