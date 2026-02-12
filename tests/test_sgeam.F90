program test_sgeam
  use gpu
  use iso_c_binding
  implicit none

  ! Test configuration
  integer, parameter :: m = 40
  integer, parameter :: n = 35
  real, parameter :: tol = 1.0e-5

  ! Handles and pointers
  type(gpu_blas) :: handle
  type(gpu_real2) :: a, b, c

  ! Host arrays for verification
  real, allocatable :: a_h(:,:), b_h(:,:), c_h(:,:), c_expected(:,:)
  real :: alpha, beta
  integer :: i, j
  integer :: test_passed, total_tests

  print *, "================================"
  print *, "SGEAM (matrix addition) Tests"
  print *, "================================"
  print *, ""

  test_passed = 0
  total_tests = 0

  ! Initialize BLAS handle
  call gpu_blas_create(handle)

  ! Allocate arrays
  allocate(a_h(m,n), b_h(m,n), c_h(m,n), c_expected(m,n))

  ! Initialize test data
  do j = 1, n
    do i = 1, m
      a_h(i,j) = real(i + j)
      b_h(i,j) = real(i * j)
    end do
  end do

  ! =========================================
  ! Test 1: C = A + B (alpha=1, beta=1, no transpose)
  ! =========================================
  print *, "Test 1: C = A + B (N, N, alpha=1, beta=1)"
  total_tests = total_tests + 1

  call gpu_allocate(a, m, n)
  call gpu_allocate(b, m, n)
  call gpu_allocate(c, m, n)

  call gpu_upload(a_h, a)
  call gpu_upload(b_h, b)
  c_h = 0.0
  call gpu_upload(c_h, c)

  alpha = 1.0
  beta = 1.0
  call gpu_sgeam(handle, 'N', 'N', m, n, alpha, a%f(1,1), m, beta, b%f(1,1), m, c%f(1,1), m)
  call gpu_download(c, c_h)

  c_expected = a_h + b_h

  if (maxval(abs(c_h - c_expected)) < tol) then
    print *, "  PASSED"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Max error:", maxval(abs(c_h - c_expected))
  end if

  ! =========================================
  ! Test 2: C = alpha * A + beta * B (alpha=2.5, beta=1.5)
  ! =========================================
  print *, ""
  print *, "Test 2: C = alpha * A + beta * B (N, N, alpha=2.5, beta=1.5)"
  total_tests = total_tests + 1

  c_h = 0.0
  call gpu_upload(c_h, c)

  alpha = 2.5
  beta = 1.5
  call gpu_sgeam(handle, 'N', 'N', m, n, alpha, a%f(1,1), m, beta, b%f(1,1), m, c%f(1,1), m)
  call gpu_download(c, c_h)

  c_expected = alpha * a_h + beta * b_h

  if (maxval(abs(c_h - c_expected)) < tol * (abs(alpha) + abs(beta)) * 10.0) then
    print *, "  PASSED"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Max error:", maxval(abs(c_h - c_expected))
  end if

  ! =========================================
  ! Test 3: C = A^T + B (transpose A)
  ! =========================================
  print *, ""
  print *, "Test 3: C = A^T + B (T, N, alpha=1, beta=1)"
  total_tests = total_tests + 1

  call gpu_deallocate(a)
  call gpu_deallocate(b)
  call gpu_deallocate(c)

  deallocate(a_h, b_h, c_h, c_expected)
  allocate(a_h(n,m), b_h(m,n), c_h(m,n), c_expected(m,n))

  do j = 1, m
    do i = 1, n
      a_h(i,j) = real(i + j)
    end do
  end do

  do j = 1, n
    do i = 1, m
      b_h(i,j) = real(i * j)
    end do
  end do

  call gpu_allocate(a, n, m)
  call gpu_allocate(b, m, n)
  call gpu_allocate(c, m, n)

  call gpu_upload(a_h, a)
  call gpu_upload(b_h, b)
  c_h = 0.0
  call gpu_upload(c_h, c)

  alpha = 1.0
  beta = 1.0
  call gpu_sgeam(handle, 'T', 'N', m, n, alpha, a%f(1,1), n, beta, b%f(1,1), m, c%f(1,1), m)
  call gpu_download(c, c_h)

  c_expected = transpose(a_h) + b_h

  if (maxval(abs(c_h - c_expected)) < tol * 10.0) then
    print *, "  PASSED"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Max error:", maxval(abs(c_h - c_expected))
  end if

  ! =========================================
  ! Test 4: C = A + B^T (transpose B)
  ! =========================================
  print *, ""
  print *, "Test 4: C = A + B^T (N, T, alpha=1, beta=1)"
  total_tests = total_tests + 1

  call gpu_deallocate(a)
  call gpu_deallocate(b)
  call gpu_deallocate(c)

  deallocate(a_h, b_h, c_h, c_expected)
  allocate(a_h(m,n), b_h(n,m), c_h(m,n), c_expected(m,n))

  do j = 1, n
    do i = 1, m
      a_h(i,j) = real(i + j)
    end do
  end do

  do j = 1, m
    do i = 1, n
      b_h(i,j) = real(i * j)
    end do
  end do

  call gpu_allocate(a, m, n)
  call gpu_allocate(b, n, m)
  call gpu_allocate(c, m, n)

  call gpu_upload(a_h, a)
  call gpu_upload(b_h, b)
  c_h = 0.0
  call gpu_upload(c_h, c)

  alpha = 1.0
  beta = 1.0
  call gpu_sgeam(handle, 'N', 'T', m, n, alpha, a%f(1,1), m, beta, b%f(1,1), n, c%f(1,1), m)
  call gpu_download(c, c_h)

  c_expected = a_h + transpose(b_h)

  if (maxval(abs(c_h - c_expected)) < tol * 10.0) then
    print *, "  PASSED"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Max error:", maxval(abs(c_h - c_expected))
  end if

  ! Cleanup
  call gpu_deallocate(a)
  call gpu_deallocate(b)
  call gpu_deallocate(c)
  call gpu_blas_destroy(handle)
  deallocate(a_h, b_h, c_h, c_expected)

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

end program test_sgeam
