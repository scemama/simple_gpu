AC_PREREQ([2.69])

AC_INIT([simple_gpu], [1.0], [https://github.com/scemama/simple_gpu], [], [https://github.com/scemama/simple_gpu])
# Author: Anthony Scemama
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_AUX_DIR([build-config])

AM_INIT_AUTOMAKE([foreign -Wall -Werror color-tests parallel-tests silent-rules subdir-objects])
AC_PROG_F77
AC_PROG_CC
AC_PROG_FC
AM_PROG_AR
LT_INIT([shared static])


## BLAS
AX_BLAS([], [AC_MSG_ERROR([BLAS was not found.])])

AC_SUBST([BLAS_LIBS])

# Option to disable NVIDIA library
AC_ARG_ENABLE([nvidia],
  [AS_HELP_STRING([--disable-nvidia], [disable NVIDIA GPU library])],
  [enable_nvidia=$enableval],
  [enable_nvidia=yes])

# Optional CUDA installation directory
AC_ARG_WITH([cuda],
  [AS_HELP_STRING([--with-cuda=DIR], [CUDA installation directory])],
  [CUDA_PATH="$withval"],
  [CUDA_PATH=""])

# Option to disable AMD library
AC_ARG_ENABLE([amd],
  [AS_HELP_STRING([--disable-amd], [disable AMD GPU library])],
  [enable_amd=$enableval],
  [enable_amd=yes])

# Optional ROCm installation directory
AC_ARG_WITH([rocm],
  [AS_HELP_STRING([--with-rocm=DIR], [ROCm installation directory])],
  [ROCM_PATH="$withval"],
  [ROCM_PATH=""])

# Check for CUDA/cuBLAS libraries if NVIDIA is enabled
BUILD_NVIDIA=no
if test "x$enable_nvidia" = "xyes"; then

  # Set up CUDA flags if path is provided
  if test -n "$CUDA_PATH"; then
    CUDA_CFLAGS="-I$CUDA_PATH/include"
    CUDA_LDFLAGS="-L$CUDA_PATH/lib64 -L$CUDA_PATH/lib"
  else
    CUDA_CFLAGS=""
    CUDA_LDFLAGS=""
  fi

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

  if test "x$HAVE_CUDART" = "xyes" && test "x$HAVE_CUBLAS" = "xyes"; then
    BUILD_NVIDIA=yes
    CUDA_LIBS="-lcudart -lcublas"
    AC_SUBST([CUDA_CFLAGS])
    AC_SUBST([CUDA_LDFLAGS])
    AC_SUBST([CUDA_LIBS])
    AC_SUBST([CUDA_PATH])
  else
    AC_MSG_WARN([CUDA libraries not found. NVIDIA GPU library will not be built.])
  fi
fi

AM_CONDITIONAL([BUILD_NVIDIA], [test "x$BUILD_NVIDIA" = "xyes"])

# Check for ROCm/HIP libraries if AMD is enabled
BUILD_AMD=no
if test "x$enable_amd" = "xyes"; then

  # Set up ROCm flags if path is provided
  if test -n "$ROCM_PATH"; then
    ROCM_CFLAGS="-I$ROCM_PATH/include"
    ROCM_LDFLAGS="-L$ROCM_PATH/lib"
  else
    ROCM_CFLAGS=""
    ROCM_LDFLAGS=""
  fi

  # Save current flags
  OLD_CFLAGS="$CFLAGS"
  OLD_LDFLAGS="$LDFLAGS"
  CFLAGS="$CFLAGS $ROCM_CFLAGS"
  LDFLAGS="$LDFLAGS $ROCM_LDFLAGS"

  # Check for HIP runtime library
  AC_CHECK_LIB([amdhip64], [hipMalloc], [HAVE_HIP=yes], [HAVE_HIP=no])

  # Check for hipBLAS library
  AC_CHECK_LIB([hipblas], [hipblasDgemm], [HAVE_HIPBLAS=yes], [HAVE_HIPBLAS=no], [-lamdhip64])

  # Restore flags
  CFLAGS="$OLD_CFLAGS"
  LDFLAGS="$OLD_LDFLAGS"

  if test "x$HAVE_HIP" = "xyes" && test "x$HAVE_HIPBLAS" = "xyes"; then
    BUILD_AMD=yes
    ROCM_LIBS="-lamdhip64 -lhipblas"
    AC_SUBST([ROCM_CFLAGS])
    AC_SUBST([ROCM_LDFLAGS])
    AC_SUBST([ROCM_LIBS])
    AC_SUBST([ROCM_PATH])
  else
    AC_MSG_WARN([ROCm libraries not found. AMD GPU library will not be built.])
  fi
fi

AM_CONDITIONAL([BUILD_AMD], [test "x$BUILD_AMD" = "xyes"])

# Output files
AC_CONFIG_FILES([
  Makefile
  pkgconfig/simple_gpu-cpu.pc
  pkgconfig/simple_gpu-nvidia.pc
  pkgconfig/simple_gpu-amd.pc
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
  echo "  CUDA_PATH:       $CUDA_PATH"
  echo "  CUDA_CFLAGS:     $CUDA_CFLAGS"
  echo "  CUDA_LDFLAGS:    $CUDA_LDFLAGS"
  echo "  CUDA_LIBS:       $CUDA_LIBS"
else
  echo "  NVIDIA GPU:      disabled"
fi
echo ""
if test "x$BUILD_AMD" = "xyes"; then
  echo "  AMD GPU:         enabled"
  echo "  ROCM_PATH:       $ROCM_PATH"
  echo "  ROCM_CFLAGS:     $ROCM_CFLAGS"
  echo "  ROCM_LDFLAGS:    $ROCM_LDFLAGS"
  echo "  ROCM_LIBS:       $ROCM_LIBS"
else
  echo "  AMD GPU:         disabled"
fi
echo "========================================"
echo ""
