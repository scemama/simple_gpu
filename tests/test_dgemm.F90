program test_dgemm
  use gpu
  use iso_c_binding
  implicit none

  ! Test configuration
  integer, parameter :: m = 30
  integer, parameter :: n = 25
  integer, parameter :: k = 20
  double precision, parameter :: tol = 1.0d-8

  ! Handles and pointers
  type(gpu_blas) :: handle
  type(gpu_double2) :: a, b, c

  ! Host arrays for verification
  double precision, allocatable :: a_h(:,:), b_h(:,:), c_h(:,:), c_expected(:,:)
  double precision :: alpha, beta
  integer :: i, j
  integer :: test_passed, total_tests

  print *, "================================"
  print *, "DGEMM (matrix-matrix) Tests"
  print *, "================================"
  print *, ""

  test_passed = 0
  total_tests = 0

  ! Initialize BLAS handle
  call gpu_blas_create(handle)

  ! Allocate arrays
  allocate(a_h(m,k), b_h(k,n), c_h(m,n), c_expected(m,n))

  ! Initialize test data
  do j = 1, k
    do i = 1, m
      a_h(i,j) = dble(i + j)
    end do
  end do

  do j = 1, n
    do i = 1, k
      b_h(i,j) = dble(i * j)
    end do
  end do

  ! =========================================
  ! Test 1: C = A * B (alpha=1, beta=0, no transpose)
  ! =========================================
  print *, "Test 1: C = A * B (N, N, alpha=1, beta=0)"
  total_tests = total_tests + 1

  call gpu_allocate(a, m, k)
  call gpu_allocate(b, k, n)
  call gpu_allocate(c, m, n)

  call gpu_upload(a_h, a)
  call gpu_upload(b_h, b)
  c_h = 0.0d0
  call gpu_upload(c_h, c)

  alpha = 1.0d0
  beta = 0.0d0
  call gpu_dgemm(handle, 'N', 'N', m, n, k, alpha, a%f(1,1), m, b%f(1,1), k, beta, c%f(1,1), m)
  call gpu_download(c, c_h)

  c_expected = matmul(a_h, b_h)

  if (maxval(abs(c_h - c_expected)) < tol * k) then
    print *, "  PASSED"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Max error:", maxval(abs(c_h - c_expected))
  end if

  ! =========================================
  ! Test 2: C = alpha * A * B (alpha=2.5, beta=0, no transpose)
  ! =========================================
  print *, ""
  print *, "Test 2: C = alpha * A * B (N, N, alpha=2.5, beta=0)"
  total_tests = total_tests + 1

  c_h = 0.0d0
  call gpu_upload(c_h, c)

  alpha = 2.5d0
  beta = 0.0d0
  call gpu_dgemm(handle, 'N', 'N', m, n, k, alpha, a%f(1,1), m, b%f(1,1), k, beta, c%f(1,1), m)
  call gpu_download(c, c_h)

  c_expected = alpha * matmul(a_h, b_h)

  if (maxval(abs(c_h - c_expected)) < tol * k * abs(alpha)) then
    print *, "  PASSED"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Max error:", maxval(abs(c_h - c_expected))
  end if

  ! =========================================
  ! Test 3: C = alpha * A * B + beta * C (alpha=1.5, beta=0.5)
  ! =========================================
  print *, ""
  print *, "Test 3: C = alpha * A * B + beta * C (N, N, alpha=1.5, beta=0.5)"
  total_tests = total_tests + 1

  do j = 1, n
    do i = 1, m
      c_h(i,j) = dble(i + j) * 2.0d0
    end do
  end do
  call gpu_upload(c_h, c)

  alpha = 1.5d0
  beta = 0.5d0
  c_expected = c_h
  call gpu_dgemm(handle, 'N', 'N', m, n, k, alpha, a%f(1,1), m, b%f(1,1), k, beta, c%f(1,1), m)
  call gpu_download(c, c_h)

  c_expected = alpha * matmul(a_h, b_h) + beta * c_expected

  if (maxval(abs(c_h - c_expected)) < tol * k * (abs(alpha) + abs(beta)) * 10.0d0) then
    print *, "  PASSED"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Max error:", maxval(abs(c_h - c_expected))
  end if

  ! =========================================
  ! Test 4: C = A^T * B (transpose A)
  ! =========================================
  print *, ""
  print *, "Test 4: C = A^T * B (T, N, alpha=1, beta=0)"
  total_tests = total_tests + 1

  call gpu_deallocate(a)
  call gpu_deallocate(b)
  call gpu_deallocate(c)

  deallocate(a_h, b_h, c_h, c_expected)
  allocate(a_h(k,m), b_h(k,n), c_h(m,n), c_expected(m,n))

  do j = 1, m
    do i = 1, k
      a_h(i,j) = dble(i + j)
    end do
  end do

  do j = 1, n
    do i = 1, k
      b_h(i,j) = dble(i * j)
    end do
  end do

  call gpu_allocate(a, k, m)
  call gpu_allocate(b, k, n)
  call gpu_allocate(c, m, n)

  call gpu_upload(a_h, a)
  call gpu_upload(b_h, b)
  c_h = 0.0d0
  call gpu_upload(c_h, c)

  alpha = 1.0d0
  beta = 0.0d0
  call gpu_dgemm(handle, 'T', 'N', m, n, k, alpha, a%f(1,1), k, b%f(1,1), k, beta, c%f(1,1), m)
  call gpu_download(c, c_h)

  c_expected = matmul(transpose(a_h), b_h)

  if (maxval(abs(c_h - c_expected)) < tol * k) then
    print *, "  PASSED"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Max error:", maxval(abs(c_h - c_expected))
  end if

  ! =========================================
  ! Test 5: C = A * B^T (transpose B)
  ! =========================================
  print *, ""
  print *, "Test 5: C = A * B^T (N, T, alpha=1, beta=0)"
  total_tests = total_tests + 1

  call gpu_deallocate(a)
  call gpu_deallocate(b)
  call gpu_deallocate(c)

  deallocate(a_h, b_h, c_h, c_expected)
  allocate(a_h(m,k), b_h(n,k), c_h(m,n), c_expected(m,n))

  do j = 1, k
    do i = 1, m
      a_h(i,j) = dble(i + j)
    end do
  end do

  do j = 1, k
    do i = 1, n
      b_h(i,j) = dble(i * j)
    end do
  end do

  call gpu_allocate(a, m, k)
  call gpu_allocate(b, n, k)
  call gpu_allocate(c, m, n)

  call gpu_upload(a_h, a)
  call gpu_upload(b_h, b)
  c_h = 0.0d0
  call gpu_upload(c_h, c)

  alpha = 1.0d0
  beta = 0.0d0
  call gpu_dgemm(handle, 'N', 'T', m, n, k, alpha, a%f(1,1), m, b%f(1,1), n, beta, c%f(1,1), m)
  call gpu_download(c, c_h)

  c_expected = matmul(a_h, transpose(b_h))

  if (maxval(abs(c_h - c_expected)) < tol * k) then
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

end program test_dgemm
