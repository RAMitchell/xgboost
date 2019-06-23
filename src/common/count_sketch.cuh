/*!
 * Copyright 2019 by Contributors
 * \file  count_sketch.cuh
 */
#pragma once
#include <thrust/device_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>

template <typename KeyT>
__device__ size_t h(KeyT key, size_t w, size_t seed) {
  return ((key * 179430413) + seed * 179426453) % w;
}

template <typename KeyT, typename CountT>
__device__ CountT g(KeyT key, CountT count, size_t seed) {
  size_t hash = (key * 2654435761) + seed * 1301011 % 2;
  if (hash == 0) {
    return count;
  } else {
    return -count;
  }
}

template <typename KeyT, typename CountT,
          typename AllocT = thrust::device_allocator<CountT>>
class CountSketch {
  using StorageVectorT = thrust::device_vector<CountT, AllocT>;
  StorageVectorT data;
  const size_t w;
  const size_t d;

 public:
  CountSketch(size_t w, size_t d) : w(w), d(d) { data.resize(w * d); }
  template <typename KeyIteratorT, typename CountIteratorT>
  void Add(KeyIteratorT keys_begin, KeyIteratorT keys_end,
           CountIteratorT counts_begin) {
    static_assert(
        std::is_same<typename std::iterator_traits<KeyIteratorT>::value_type,
                     KeyT>::value,
        "Key iterator value type must be KeyT");
    static_assert(
        std::is_same<typename std::iterator_traits<CountIteratorT>::value_type,
                     CountT>::value,
        "Count iterator value type must be CountT");
    auto d_copy = this->d;
    auto w_copy = this->w;
    auto d_data = this->data.data().get();
    auto counting = thrust::counting_iterator<size_t>(0);
    size_t n = keys_end - keys_begin;
    thrust::for_each_n(counting, n, [=] __device__(size_t idx) {
      for (auto i = 0ull; i < d_copy; i++) {
        auto k = h(static_cast<KeyT>(keys_begin[idx]), w_copy, i);
        auto val=g(keys_begin[idx], static_cast<CountT>(counts_begin[idx]), i);
        atomicAdd(&d_data[i * w_copy + k], val);
      }
    });
  }
  template <typename KeyIteratorT>
  void Add(KeyIteratorT keys_begin, KeyIteratorT keys_end
    ) {
    this->Add(keys_begin, keys_end,
              thrust::constant_iterator<CountT>(CountT(1)));
  }
  template <typename KeyIteratorT, typename CountIteratorT>
  void PointQuery(KeyIteratorT begin, KeyIteratorT end, CountIteratorT out) {
    using ValueT = typename std::iterator_traits<KeyIteratorT>::value_type;
    static_assert(std::is_same<ValueT, KeyT>::value,
                  "Iterator value type must be KeyT");
    auto d_copy = this->d;
    auto w_copy = this->w;
    auto d_data = this->data.data().get();
    auto counting = thrust::counting_iterator<size_t>(0);
    size_t n = end - begin;
    StorageVectorT temp(n * this->d);
    auto d_temp = temp.data().get();
    thrust::for_each_n(counting, n, [=] __device__(size_t idx) {
      KeyT key = begin[idx];
      CountT* local_temp = d_temp + d_copy * idx;
      for (auto i = 0ull; i < d_copy; i++) {
        auto k = h(key, w_copy, i);
        local_temp[i] = g(key, static_cast<CountT>(d_data[i * w_copy + k]), i);
      }
      // Return median
      thrust::sort(thrust::seq, local_temp, local_temp + d_copy);
      out[idx] = local_temp[d_copy / 2];
    });
  }
};
