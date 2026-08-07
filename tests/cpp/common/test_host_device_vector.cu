/**
 * Copyright 2018-2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <xgboost/host_device_vector.h>

#include "../../../src/common/cuda_rt_utils.h"  // for SetDevice
#include "../../../src/common/device_helpers.cuh"

namespace xgboost::common {
namespace {
void SetDeviceForTest(DeviceOrd device) {
  int n_devices;
  dh::safe_cuda(cudaGetDeviceCount(&n_devices));
  device.ordinal %= n_devices;
  dh::safe_cuda(cudaSetDevice(device.ordinal));
}
}  // namespace

struct HostDeviceVectorSetDeviceHandler {
  template <typename Functor>
  explicit HostDeviceVectorSetDeviceHandler(Functor f) {
    SetCudaSetDeviceHandler(f);
  }

  ~HostDeviceVectorSetDeviceHandler() { SetCudaSetDeviceHandler(nullptr); }
};

void InitHostDeviceVector(size_t n, DeviceOrd device, HostDeviceVector<int>* v) {
  // create the vector
  v->Resize(n, device);

  ASSERT_EQ(v->Size(), n);
  ASSERT_EQ(v->Device(), device);
  // ensure that the device have read-write access
  ASSERT_TRUE(v->DeviceCanRead());
  ASSERT_TRUE(v->DeviceCanWrite());
  // ensure that the host has no access
  ASSERT_FALSE(v->HostCanRead());
  ASSERT_FALSE(v->HostCanWrite());

  // fill in the data on the host
  std::vector<int>& data_h = v->HostVector();
  // ensure that the host has full access, while the device have none
  ASSERT_TRUE(v->HostCanRead());
  ASSERT_TRUE(v->HostCanWrite());
  ASSERT_FALSE(v->DeviceCanRead());
  ASSERT_FALSE(v->DeviceCanWrite());
  ASSERT_EQ(data_h.size(), n);
  std::copy_n(thrust::make_counting_iterator(0), n, data_h.begin());
}

void PlusOne(HostDeviceVector<int>* v) {
  auto device = v->Device();
  SetDeviceForTest(device);
  thrust::transform(dh::tcbegin(*v), dh::tcend(*v), dh::tbegin(*v),
                    [=] __device__(unsigned int a) { return a + 1; });
  ASSERT_TRUE(v->DeviceCanWrite());
}

void CheckDevice(HostDeviceVector<int>* v, size_t size, unsigned int first, GPUAccess access) {
  ASSERT_EQ(v->Size(), size);
  SetDeviceForTest(v->Device());

  ASSERT_TRUE(thrust::equal(dh::tcbegin(*v), dh::tcend(*v), thrust::make_counting_iterator(first)));
  ASSERT_TRUE(v->DeviceCanRead());
  // ensure that the device has at most the access specified by access
  ASSERT_EQ(v->DeviceCanWrite(), access == GPUAccess::kWrite);
  ASSERT_EQ(v->HostCanRead(), access == GPUAccess::kRead);
  ASSERT_FALSE(v->HostCanWrite());

  ASSERT_TRUE(thrust::equal(dh::tbegin(*v), dh::tend(*v), thrust::make_counting_iterator(first)));
  ASSERT_TRUE(v->DeviceCanRead());
  ASSERT_TRUE(v->DeviceCanWrite());
  ASSERT_FALSE(v->HostCanRead());
  ASSERT_FALSE(v->HostCanWrite());
}

void CheckHost(HostDeviceVector<int>* v, GPUAccess access) {
  const std::vector<int>& data_h =
      access == GPUAccess::kNone ? v->HostVector() : v->ConstHostVector();
  for (size_t i = 0; i < v->Size(); ++i) {
    ASSERT_EQ(data_h.at(i), i + 1);
  }
  ASSERT_TRUE(v->HostCanRead());
  ASSERT_EQ(v->HostCanWrite(), access == GPUAccess::kNone);
  ASSERT_EQ(v->DeviceCanRead(), access == GPUAccess::kRead);
  // the devices should have no write access
  ASSERT_FALSE(v->DeviceCanWrite());
}

void TestHostDeviceVector(size_t n, DeviceOrd device) {
  HostDeviceVectorSetDeviceHandler hdvec_dev_hndlr(curt::SetDevice);
  HostDeviceVector<int> v;
  InitHostDeviceVector(n, device, &v);
  CheckDevice(&v, n, 0, GPUAccess::kRead);
  PlusOne(&v);
  CheckDevice(&v, n, 1, GPUAccess::kWrite);
  CheckHost(&v, GPUAccess::kRead);
  CheckHost(&v, GPUAccess::kNone);
}

TEST(HostDeviceVector, Basic) {
  size_t n = 1001;
  DeviceOrd device = DeviceOrd::CUDA(0);
  TestHostDeviceVector(n, device);
}

TEST(HostDeviceVector, Copy) {
  size_t n = 1001;
  auto device = DeviceOrd::CUDA(0);
  HostDeviceVectorSetDeviceHandler hdvec_dev_hndlr(curt::SetDevice);

  HostDeviceVector<int> v;
  {
    // a separate scope to ensure that v1 is gone before further checks
    HostDeviceVector<int> v1;
    InitHostDeviceVector(n, device, &v1);
    v.Resize(v1.Size(), device);
    v.Copy(v1);
  }
  CheckDevice(&v, n, 0, GPUAccess::kWrite);
  PlusOne(&v);
  CheckDevice(&v, n, 1, GPUAccess::kWrite);
  CheckHost(&v, GPUAccess::kRead);
  CheckHost(&v, GPUAccess::kNone);

  HostDeviceVector<int> device_source;
  InitHostDeviceVector(n, device, &device_source);
  device_source.DeviceSpan(device);
  HostDeviceVector<int> host_destination(n);
  host_destination.Copy(device_source);
  ASSERT_EQ(host_destination.Device(), DeviceOrd::CPU());
  ASSERT_TRUE(host_destination.HostCanWrite());
  auto const& copied = host_destination.ConstHostVector();
  for (std::size_t i = 0; i < copied.size(); ++i) {
    ASSERT_EQ(copied[i], i);
  }
}

TEST(HostDeviceVector, Extend) {
  auto device = DeviceOrd::CUDA(0);
  HostDeviceVector<int> lhs{0, 1};
  HostDeviceVector<int> rhs{2, 3};
  lhs.Extend(rhs, device);
  ASSERT_EQ(lhs.Device(), device);
  ASSERT_TRUE(lhs.DeviceCanWrite());
  ASSERT_EQ(lhs.ConstHostVector(), std::vector<int>({0, 1, 2, 3}));

  rhs.DeviceSpan(device);
  HostDeviceVector<int> host_lhs{0, 1};
  host_lhs.Extend(rhs, DeviceOrd::CPU());
  ASSERT_EQ(host_lhs.Device(), DeviceOrd::CPU());
  ASSERT_TRUE(host_lhs.HostCanWrite());
  ASSERT_EQ(host_lhs.ConstHostVector(), std::vector<int>({0, 1, 2, 3}));
}

TEST(HostDeviceVector, Span) {
  HostDeviceVector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};
  auto span = vec.DeviceSpan(DeviceOrd::CUDA(0));
  ASSERT_EQ(vec.Device(), DeviceOrd::CUDA(0));
  ASSERT_EQ(vec.Size(), span.size());
  ASSERT_EQ(vec.DevicePointer(), span.data());
  auto const_span = vec.ConstDeviceSpan(DeviceOrd::CUDA(0));
  ASSERT_EQ(vec.Size(), const_span.size());
  ASSERT_EQ(vec.ConstDevicePointer(), const_span.data());

  auto h_span = vec.ConstHostSpan();
  ASSERT_TRUE(vec.HostCanRead());
  ASSERT_FALSE(vec.HostCanWrite());
  ASSERT_EQ(h_span.size(), vec.Size());
  ASSERT_EQ(h_span.data(), vec.ConstHostPointer());

  h_span = vec.HostSpan();
  ASSERT_TRUE(vec.HostCanWrite());
}

TEST(HostDeviceVector, Empty) {
  HostDeviceVector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};
  HostDeviceVector<float> another{std::move(vec)};
  ASSERT_FALSE(another.Empty());
  ASSERT_TRUE(vec.Empty());
}

TEST(HostDeviceVector, Resize) {
  auto check = [&](HostDeviceVector<float> const& vec) {
    auto const& h_vec = vec.ConstHostSpan();
    for (std::size_t i = 0; i < 4; ++i) {
      ASSERT_EQ(h_vec[i], i + 1);
    }
    for (std::size_t i = 4; i < vec.Size(); ++i) {
      ASSERT_EQ(h_vec[i], 3.0);
    }
  };
  {
    HostDeviceVector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};
    vec.ConstDeviceSpan(DeviceOrd::CUDA(0));
    ASSERT_TRUE(vec.DeviceCanRead());
    ASSERT_FALSE(vec.DeviceCanWrite());
    vec.DeviceSpan(vec.Device());
    vec.Resize(7, 3.0f);
    ASSERT_TRUE(vec.DeviceCanWrite());
    check(vec);
  }
  {
    HostDeviceVector<float> vec{{1.0f, 2.0f, 3.0f, 4.0f}, DeviceOrd::CUDA(0)};
    ASSERT_TRUE(vec.DeviceCanWrite());
    vec.Resize(7, 3.0f);
    ASSERT_TRUE(vec.DeviceCanWrite());
    check(vec);
  }
  {
    HostDeviceVector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_TRUE(vec.HostCanWrite());
    vec.Resize(7, 3.0f);
    ASSERT_TRUE(vec.HostCanWrite());
    check(vec);
  }
}
}  // namespace xgboost::common
