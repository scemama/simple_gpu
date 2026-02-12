program test_cpu_basic
  use gpu
  use iso_c_binding
  implicit none
  
  ! Test configuration
  integer, parameter :: n = 100
  integer, parameter :: m = 50
  double precision, parameter :: tol = 1.0d-10
  
  ! Handles and pointers
  type(gpu_blas) :: handle
  type(gpu_double1) :: x, y
  type(gpu_double2) :: a, b, c
  
  ! Host arrays for verification
  double precision, allocatable :: x_h(:), y_h(:)
  double precision, allocatable :: a_h(:,:), b_h(:,:), c_h(:,:)
  
  ! Results
  double precision :: result
  double precision :: alpha, beta
  integer :: i, j
  integer :: test_passed, total_tests
  
  print *, "================================"
  print *, "GPU CPU Library Basic Tests"
  print *, "================================"
  print *, ""
  
  test_passed = 0
  total_tests = 0
  
  ! Initialize data
  allocate(x_h(n), y_h(n))
  allocate(a_h(m,n), b_h(m,n), c_h(m,n))
  
  ! Fill with test data
  do i = 1, n
    x_h(i) = dble(i)
    y_h(i) = dble(i) * 2.0d0
  end do
  
  do j = 1, n
    do i = 1, m
      a_h(i,j) = dble(i + j)
      b_h(i,j) = dble(i * j)
    end do
  end do
  
  alpha = 1.5d0
  beta = 0.5d0
  
  ! =========================================
  ! Test 1: DOT product
  ! =========================================
  print *, "Test 1: DDOT (dot product)"
  total_tests = total_tests + 1
  
  call gpu_blas_create(handle)
  call gpu_allocate(x, n)
  call gpu_allocate(y, n)
  
  ! Upload data
  call gpu_upload(x_h, x)
  call gpu_upload(y_h, y)
  
  call gpu_ddot(handle, n, x%f(1), 1, y%f(1), 1, result)
  
  ! Verify result
  if (abs(result - sum(x_h * y_h)) < tol) then
    print *, "  PASSED - Result correct:", result
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Result incorrect"
    print *, "    Expected:", sum(x_h * y_h)
    print *, "    Got:     ", result
  end if
  
  call gpu_deallocate(x)
  call gpu_deallocate(y)
  call gpu_blas_destroy(handle)
  
  ! =========================================
  ! Test 2: GEAM (matrix addition)
  ! =========================================
  print *, ""
  print *, "Test 2: DGEAM (matrix addition)"
  total_tests = total_tests + 1
  
  call gpu_blas_create(handle)
  call gpu_allocate(a, m, n)
  call gpu_allocate(b, m, n)
  call gpu_allocate(c, m, n)
  
  ! Upload data
  call gpu_upload(a_h, a)
  call gpu_upload(b_h, b)
  c_h = 0.0d0
  call gpu_upload(c_h, c)
  
  ! C = alpha * A + beta * B
  call gpu_dgeam(handle, 'N', 'N', m, n, &
                 alpha, a%f(1,1), m, beta, b%f(1,1), m, &
                 c%f(1,1), m)
  
  call gpu_download(c, c_h)
  
  ! Verify result
  if (maxval(abs(c_h - (alpha * a_h + beta * b_h))) < tol) then
    print *, "  PASSED - Result correct"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Result incorrect"
    print *, "    Max error:", maxval(abs(c_h - (alpha * a_h + beta * b_h)))
  end if
  
  call gpu_deallocate(a)
  call gpu_deallocate(b)
  call gpu_deallocate(c)
  call gpu_blas_destroy(handle)
  
  ! =========================================
  ! Test 3: GEMV (matrix-vector multiply)
  ! =========================================
  print *, ""
  print *, "Test 3: DGEMV (matrix-vector multiply)"
  total_tests = total_tests + 1
  
  call gpu_blas_create(handle)
  call gpu_allocate(a, m, n)
  call gpu_allocate(x, n)
  call gpu_allocate(y, m)
  
  ! Upload data
  call gpu_upload(a_h, a)
  call gpu_upload(x_h, x)
  y_h(1:m) = 0.0d0
  call gpu_upload(y_h(1:m), y)
  
  ! y = alpha * A * x + beta * y
  alpha = 1.0d0
  beta = 0.0d0
  call gpu_dgemv(handle, 'N', m, n, &
                 alpha, a%f(1,1), m, x%f(1), 1, beta, y%f(1), 1)
  
  call gpu_download(y, y_h(1:m))
  
  ! Verify result: y = A * x
  y_h(1:m) = 0.0d0
  do j = 1, n
    do i = 1, m
      y_h(i) = y_h(i) + a_h(i,j) * x_h(j)
    end do
  end do
  
  if (maxval(abs(y%f(1:m) - y_h(1:m))) < tol * n) then
    print *, "  PASSED - Result correct"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Result incorrect"
    print *, "    Max error:", maxval(abs(y%f(1:m) - y_h(1:m)))
  end if
  
  call gpu_deallocate(a)
  call gpu_deallocate(x)
  call gpu_deallocate(y)
  call gpu_blas_destroy(handle)
  
  ! =========================================
  ! Summary
  ! =========================================
  print *, ""
  print *, "================================"
  print *, "Test Summary"
  print *, "================================"
  write(*,'(A,I0,A,I0,A)') " Passed: ", test_passed, " / ", total_tests, " tests"
  
  ! Cleanup
  deallocate(x_h, y_h, a_h, b_h, c_h)
  
  if (test_passed == total_tests) then
    print *, " Result: ALL TESTS PASSED"
    print *, "================================"
    call exit(0)
  else
    print *, " Result: SOME TESTS FAILED"
    print *, "================================"
    call exit(1)
  end if
  
end program test_cpu_basic
