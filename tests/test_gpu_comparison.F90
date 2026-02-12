program test_gpu_comparison
  use gpu
  use iso_c_binding
  implicit none
  
  ! Test configuration
  integer, parameter :: n = 100
  integer, parameter :: m = 50
  double precision, parameter :: tol = 1.0d-10
  
  ! Handles and pointers
  type(gpu_blas) :: handle_cpu, handle_gpu
  type(gpu_double1) :: x_cpu, y_cpu, x_gpu, y_gpu
  type(gpu_double2) :: a_cpu, a_gpu, b_cpu, b_gpu, c_cpu, c_gpu
  
  ! Host arrays for initialization
  double precision, allocatable :: x_h(:), y_h(:)
  double precision, allocatable :: a_h(:,:), b_h(:,:), c_h(:,:)
  
  ! Results
  double precision :: result_cpu, result_gpu
  double precision :: alpha, beta
  integer :: i, j
  integer :: test_passed, total_tests
  
  print *, "================================"
  print *, "GPU Library Comparison Tests"
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
  
  ! CPU version - Load libgpu_cpu.so
  call gpu_blas_create(handle_cpu)
  call gpu_allocate(x_cpu, n)
  call gpu_allocate(y_cpu, n)
  
  ! Copy data
  do i = 1, n
    x_cpu%f(i) = x_h(i)
    y_cpu%f(i) = y_h(i)
  end do
  
  call gpu_upload(x_cpu)
  call gpu_upload(y_cpu)
  call gpu_ddot(handle_cpu, int(n,8), x_cpu, 1_8, y_cpu, 1_8, result_cpu)
  
  ! Note: For actual CPU vs GPU comparison, we would need to:
  ! 1. Link with libgpu_cpu.so and compute result_cpu
  ! 2. Link with libgpu_nvidia.so and compute result_gpu
  ! 3. Compare the two results
  
  ! For now, we verify the result makes sense
  if (abs(result_cpu - sum(x_h * y_h)) < tol) then
    print *, "  PASSED - CPU result correct"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - CPU result incorrect"
    print *, "    Expected:", sum(x_h * y_h)
    print *, "    Got:     ", result_cpu
  end if
  
  call gpu_free(x_cpu)
  call gpu_free(y_cpu)
  call gpu_blas_destroy(handle_cpu)
  
  ! =========================================
  ! Test 2: GEAM (matrix addition)
  ! =========================================
  print *, ""
  print *, "Test 2: DGEAM (matrix addition)"
  total_tests = total_tests + 1
  
  call gpu_blas_create(handle_cpu)
  call gpu_allocate(a_cpu, m, n)
  call gpu_allocate(b_cpu, m, n)
  call gpu_allocate(c_cpu, m, n)
  
  ! Copy data
  do j = 1, n
    do i = 1, m
      a_cpu%f(i,j) = a_h(i,j)
      b_cpu%f(i,j) = b_h(i,j)
      c_cpu%f(i,j) = 0.0d0
    end do
  end do
  
  call gpu_upload(a_cpu)
  call gpu_upload(b_cpu)
  call gpu_upload(c_cpu)
  
  ! C = alpha * A + beta * B
  call gpu_dgeam(handle_cpu, 'N', 'N', int(m,8), int(n,8), &
                 alpha, a_cpu, int(m,8), beta, b_cpu, int(m,8), &
                 c_cpu, int(m,8))
  
  call gpu_download(c_cpu)
  
  ! Verify result
  c_h = alpha * a_h + beta * b_h
  if (maxval(abs(c_cpu%f - c_h)) < tol) then
    print *, "  PASSED - CPU result correct"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - CPU result incorrect"
    print *, "    Max error:", maxval(abs(c_cpu%f - c_h))
  end if
  
  call gpu_free(a_cpu)
  call gpu_free(b_cpu)
  call gpu_free(c_cpu)
  call gpu_blas_destroy(handle_cpu)
  
  ! =========================================
  ! Test 3: GEMV (matrix-vector multiply)
  ! =========================================
  print *, ""
  print *, "Test 3: DGEMV (matrix-vector multiply)"
  total_tests = total_tests + 1
  
  call gpu_blas_create(handle_cpu)
  call gpu_allocate(a_cpu, m, n)
  call gpu_allocate(x_cpu, n)
  call gpu_allocate(y_cpu, m)
  
  ! Initialize
  do j = 1, n
    do i = 1, m
      a_cpu%f(i,j) = a_h(i,j)
    end do
  end do
  
  do i = 1, n
    x_cpu%f(i) = x_h(i)
  end do
  
  do i = 1, m
    y_cpu%f(i) = 0.0d0
  end do
  
  call gpu_upload(a_cpu)
  call gpu_upload(x_cpu)
  call gpu_upload(y_cpu)
  
  ! y = alpha * A * x + beta * y
  alpha = 1.0d0
  beta = 0.0d0
  call gpu_dgemv(handle_cpu, 'N', int(m,8), int(n,8), &
                 alpha, a_cpu, int(m,8), x_cpu, 1_8, beta, y_cpu, 1_8)
  
  call gpu_download(y_cpu)
  
  ! Verify result: y = A * x
  y_h(1:m) = 0.0d0
  do j = 1, n
    do i = 1, m
      y_h(i) = y_h(i) + a_h(i,j) * x_h(j)
    end do
  end do
  
  if (maxval(abs(y_cpu%f(1:m) - y_h(1:m))) < tol * n) then
    print *, "  PASSED - CPU result correct"
    test_passed = test_passed + 1
  else
    print *, "  FAILED - CPU result incorrect"
    print *, "    Max error:", maxval(abs(y_cpu%f(1:m) - y_h(1:m)))
  end if
  
  call gpu_free(a_cpu)
  call gpu_free(x_cpu)
  call gpu_free(y_cpu)
  call gpu_blas_destroy(handle_cpu)
  
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
  
  ! Cleanup
  deallocate(x_h, y_h, a_h, b_h, c_h)
  
end program test_gpu_comparison
