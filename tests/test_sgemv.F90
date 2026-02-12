program test_sgemv
  use gpu
  use iso_c_binding
  implicit none

  ! Test configuration
  integer, parameter :: m = 50
  integer, parameter :: n = 40
  real, parameter :: tol = 1.0e-4

  ! Handles and pointers
  type(gpu_blas) :: handle
  type(gpu_real2) :: a
  type(gpu_real1) :: x, y

  ! Host arrays for verification
  real, allocatable :: a_h(:,:), x_h(:), y_h(:), y_expected(:)
  real :: alpha, beta
  integer :: i, j
  integer :: test_passed, total_tests

  print *, "================================"
  print *, "SGEMV (matrix-vector) Tests"
  print *, "================================"
  print *, ""

  test_passed = 0
  total_tests = 0

  ! Initialize BLAS handle
  call gpu_blas_create(handle)

  ! Allocate arrays
  allocate(a_h(m,n), x_h(n), y_h(m), y_expected(m))

  ! Initialize test data
  do j = 1, n
    do i = 1, m
      a_h(i,j) = real(i + j)
    end do
  end do

  do i = 1, n
    x_h(i) = real(i)
  end do

  ! =========================================
  ! Test 1: y = A * x (alpha=1, beta=0, no transpose)
  ! =========================================
  print *, "Test 1: y = A * x (N, alpha=1, beta=0)"
  total_tests = total_tests + 1

  call gpu_allocate(a, m, n)
  call gpu_allocate(x, n)
  call gpu_allocate(y, m)

  call gpu_upload(a_h, a)
  call gpu_upload(x_h, x)
  y_h = 0.0
  call gpu_upload(y_h, y)

  alpha = 1.0
  beta = 0.0
  call gpu_sgemv(handle, 'N', m, n, alpha, a%f(1,1), m, x%f(1), 1, beta, y%f(1), 1)
  call gpu_download(y, y_h)

  ! Compute expected result
  y_expected = matmul(a_h, x_h)

  if (maxval(abs(y_h - y_expected)) < tol * n) then
    print *, "  PASSED"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Max error:", maxval(abs(y_h - y_expected))
  end if

  ! =========================================
  ! Test 2: y = alpha * A * x (alpha=2.5, beta=0, no transpose)
  ! =========================================
  print *, ""
  print *, "Test 2: y = alpha * A * x (N, alpha=2.5, beta=0)"
  total_tests = total_tests + 1

  y_h = 0.0
  call gpu_upload(y_h, y)

  alpha = 2.5
  beta = 0.0
  call gpu_sgemv(handle, 'N', m, n, alpha, a%f(1,1), m, x%f(1), 1, beta, y%f(1), 1)
  call gpu_download(y, y_h)

  y_expected = alpha * matmul(a_h, x_h)

  if (maxval(abs(y_h - y_expected)) < tol * n * abs(alpha)) then
    print *, "  PASSED"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Max error:", maxval(abs(y_h - y_expected))
  end if

  ! =========================================
  ! Test 3: y = A^T * x (transpose)
  ! =========================================
  print *, ""
  print *, "Test 3: y = A^T * x (T, alpha=1, beta=0)"
  total_tests = total_tests + 1

  call gpu_deallocate(x)
  call gpu_deallocate(y)
  call gpu_allocate(x, m)
  call gpu_allocate(y, n)

  deallocate(x_h, y_h, y_expected)
  allocate(x_h(m), y_h(n), y_expected(n))

  do i = 1, m
    x_h(i) = real(i)
  end do
  y_h = 0.0

  call gpu_upload(x_h, x)
  call gpu_upload(y_h, y)

  alpha = 1.0
  beta = 0.0
  call gpu_sgemv(handle, 'T', m, n, alpha, a%f(1,1), m, x%f(1), 1, beta, y%f(1), 1)
  call gpu_download(y, y_h)

  y_expected = matmul(transpose(a_h), x_h)

  if (maxval(abs(y_h - y_expected)) < tol * m) then
    print *, "  PASSED"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Max error:", maxval(abs(y_h - y_expected))
  end if

  ! Cleanup
  call gpu_deallocate(a)
  call gpu_deallocate(x)
  call gpu_deallocate(y)
  call gpu_blas_destroy(handle)
  deallocate(a_h, x_h, y_h, y_expected)

  ! =========================================
  ! Summary
  ! =========================================
  print *, ""
  print *, "================================"
  print *, "Test Summary"
  print *, "================================"
  write(*,'(A,I0,A,I0,A)') " Passed: ", test_passed, " / ", total_tests, " tests"

  if (test_passed == total_tests) then
    print *, " Result: ALL TESTS PASSED"
    print *, "================================"
    call exit(0)
  else
    print *, " Result: SOME TESTS FAILED"
    print *, "================================"
    call exit(1)
  end if

end program test_sgemv
