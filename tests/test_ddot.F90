program test_ddot
  use gpu
  use iso_c_binding
  implicit none

  ! Test configuration
  integer, parameter :: n = 100
  double precision, parameter :: tol = 1.0d-10

  ! Handles and pointers
  type(gpu_blas) :: handle
  type(gpu_double1) :: x, y

  ! Host arrays for verification
  double precision, allocatable :: x_h(:), y_h(:)

  ! Results
  double precision :: result, expected
  integer :: i
  integer :: test_passed, total_tests

  print *, "================================"
  print *, "DDOT (dot product) Tests"
  print *, "================================"
  print *, ""

  test_passed = 0
  total_tests = 0

  ! Initialize BLAS handle
  call gpu_blas_create(handle)

  ! Allocate device arrays
  call gpu_allocate(x, n)
  call gpu_allocate(y, n)

  ! Allocate host arrays
  allocate(x_h(n), y_h(n))

  ! =========================================
  ! Test 1: Basic dot product
  ! =========================================
  print *, "Test 1: Basic dot product (x=1..n, y=1..n)"
  total_tests = total_tests + 1

  do i = 1, n
    x_h(i) = dble(i)
    y_h(i) = dble(i)
  end do

  call gpu_upload(x_h, x)
  call gpu_upload(y_h, y)
  call gpu_ddot(handle, n, x%f(1), 1, y%f(1), 1, result)

  expected = sum(x_h * y_h)
  if (abs(result - expected) < tol) then
    print *, "  PASSED - Result:", result
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Expected:", expected, "Got:", result
  end if

  ! =========================================
  ! Test 2: Dot product with stride
  ! =========================================
  print *, ""
  print *, "Test 2: Dot product with stride=2"
  total_tests = total_tests + 1

  call gpu_ddot(handle, n/2, x%f(1), 2, y%f(1), 2, result)

  expected = 0.0d0
  do i = 1, n, 2
    expected = expected + x_h(i) * y_h(i)
  end do

  if (abs(result - expected) < tol * n) then
    print *, "  PASSED - Result:", result
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Expected:", expected, "Got:", result
  end if

  ! =========================================
  ! Test 3: Dot product with different values
  ! =========================================
  print *, ""
  print *, "Test 3: Different values (x=2*i, y=3*i)"
  total_tests = total_tests + 1

  do i = 1, n
    x_h(i) = 2.0d0 * dble(i)
    y_h(i) = 3.0d0 * dble(i)
  end do

  call gpu_upload(x_h, x)
  call gpu_upload(y_h, y)
  call gpu_ddot(handle, n, x%f(1), 1, y%f(1), 1, result)

  expected = sum(x_h * y_h)
  if (abs(result - expected) < tol * n * n) then
    print *, "  PASSED - Result:", result
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Expected:", expected, "Got:", result
  end if

  ! =========================================
  ! Test 4: Dot product with zeros
  ! =========================================
  print *, ""
  print *, "Test 4: One vector is zero"
  total_tests = total_tests + 1

  x_h = 0.0d0
  call gpu_upload(x_h, x)
  call gpu_ddot(handle, n, x%f(1), 1, y%f(1), 1, result)

  if (abs(result) < tol) then
    print *, "  PASSED - Result:", result
    test_passed = test_passed + 1
  else
    print *, "  FAILED - Expected: 0.0, Got:", result
  end if

  ! Cleanup
  call gpu_deallocate(x)
  call gpu_deallocate(y)
  call gpu_blas_destroy(handle)
  deallocate(x_h, y_h)

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

end program test_ddot
