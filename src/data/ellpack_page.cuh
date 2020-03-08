/*!
 * Copyright 2019 by XGBoost Contributors
 */

#ifndef XGBOOST_DATA_ELLPACK_PAGE_H_
#define XGBOOST_DATA_ELLPACK_PAGE_H_

#include <xgboost/data.h>

#include "../common/compressed_iterator.h"
#include "../common/device_helpers.cuh"
#include "../common/hist_util.h"

namespace xgboost {

// Find a gidx value for a given feature otherwise return -1 if not found
__forceinline__ __device__ int BinarySearchRow(
    bst_uint begin, bst_uint end,
    common::CompressedIterator<uint32_t> data,
    int const fidx_begin, int const fidx_end) {
  bst_uint previous_middle = UINT32_MAX;
  while (end != begin) {
    auto middle = begin + (end - begin) / 2;
    if (middle == previous_middle) {
      break;
    }
    previous_middle = middle;

    auto gidx = data[middle];

    if (gidx >= fidx_begin && gidx < fidx_end) {
      return gidx;
    } else if (gidx < fidx_begin) {
      begin = middle;
    } else {
      end = middle;
    }
  }
  // Value is missing
  return -1;
}

/** \brief Struct for accessing and manipulating an ellpack matrix on the
 * device. Does not own underlying memory and may be trivially copied into
 * kernels.*/
struct EllpackDeviceAccessor {
  /*! \brief Whether or not if the matrix is dense. */
  bool is_dense;
  /*! \brief Row length for ELLPack, equal to number of features. */
  size_t row_stride;
  //EllpackInfo info;
  size_t base_rowid{};
  size_t n_rows{};
  common::CompressedIterator<uint32_t> gidx_iter;
  /*! \brief Minimum value for each feature. Size equals to number of features. */
  common::Span<const bst_float> min_fvalue;
  /*! \brief Histogram cut pointers. Size equals to (number of features + 1). */
  common::Span<const uint32_t> feature_segments;
  /*! \brief Histogram cut values. Size equals to (bins per feature * number of features). */
  common::Span<const bst_float> gidx_fvalue_map;

  EllpackDeviceAccessor(int device, const common::HistogramCuts& cuts,
                        bool is_dense, size_t row_stride, size_t base_rowid,
                        size_t n_rows,common::CompressedIterator<uint32_t> gidx_iter)
      : is_dense(is_dense),
        row_stride(row_stride),
        base_rowid(base_rowid),
        n_rows(n_rows) ,gidx_iter(gidx_iter){
    cuts.cut_values_.SetDevice(device);
    cuts.cut_ptrs_.SetDevice(device);
    cuts.min_vals_.SetDevice(device);
    gidx_fvalue_map = cuts.cut_values_.ConstDeviceSpan();
    feature_segments = cuts.cut_ptrs_.ConstDeviceSpan();
    min_fvalue = cuts.min_vals_.ConstDeviceSpan();
  }
  // Get a matrix element, uses binary search for look up Return NaN if missing
  // Given a row index and a feature index, returns the corresponding cut value
  __device__ int32_t GetBinIndex(size_t ridx, size_t fidx) const {
    ridx -= base_rowid;
    auto row_begin = row_stride * ridx;
    auto row_end = row_begin + row_stride;
    auto gidx = -1;
    if (is_dense) {
      gidx = gidx_iter[row_begin + fidx];
    } else {
      gidx = BinarySearchRow(row_begin,
                             row_end,
                             gidx_iter,
                             feature_segments[fidx],
                             feature_segments[fidx + 1]);
    }
    return gidx;
  }
  __device__ bst_float GetFvalue(size_t ridx, size_t fidx) const {
    auto gidx = GetBinIndex(ridx, fidx);
    if (gidx == -1) {
      return nan("");
    }
    return gidx_fvalue_map[gidx];
  }

  // Check if the row id is withing range of the current batch.
  __device__ bool IsInRange(size_t row_id) const {
    return row_id >= base_rowid && row_id < base_rowid + n_rows;
  }
  /*! \brief Return the total number of symbols (total number of bins plus 1 for
   * not found). */
  size_t NumSymbols() const { return gidx_fvalue_map.size() + 1; }

  size_t NullValue() const { return gidx_fvalue_map.size(); }

  XGBOOST_DEVICE size_t NumBins() const { return gidx_fvalue_map.size(); }

  XGBOOST_DEVICE size_t NumFeatures() const { return min_fvalue.size(); }
};


class EllpackPageImpl {
 public:
  /*!
   * \brief Default constructor.
   *
   * This is used in the external memory case. An empty ELLPACK page is constructed with its content
   * set later by the reader.
   */
  EllpackPageImpl() = default;

  /*!
   * \brief Constructor from an existing EllpackInfo.
   *
   * This is used in the sampling case. The ELLPACK page is constructed from an existing EllpackInfo
   * and the given number of rows.
   */
  EllpackPageImpl(int device, const common::HistogramCuts& cuts, bool is_dense,
                  size_t row_stride, size_t n_rows);

  EllpackPageImpl(int device, const common::HistogramCuts& cuts,
                  const SparsePage& page,
                  bool is_dense,size_t row_stride);

  /*!
   * \brief Constructor from an existing DMatrix.
   *
   * This is used in the in-memory case. The ELLPACK page is constructed from an existing DMatrix
   * in CSR format.
   */
  explicit EllpackPageImpl(DMatrix* dmat, const BatchParam& parm);

  /*! \brief Copy the elements of the given ELLPACK page into this page.
   *
   * @param device The GPU device to use.
   * @param page The ELLPACK page to copy from.
   * @param offset The number of elements to skip before copying.
   * @returns The number of elements copied.
   */
  size_t Copy(int device, EllpackPageImpl* page, size_t offset);

  /*! \brief Compact the given ELLPACK page into the current page.
   *
   * @param device The GPU device to use.
   * @param page The ELLPACK page to compact from.
   * @param row_indexes Row indexes for the compacted page.
   */
  void Compact(int device, EllpackPageImpl* page, common::Span<size_t> row_indexes);


  /*! \return Number of instances in the page. */
  size_t Size() const;

  /*! \brief Set the base row id for this page. */
  void SetBaseRowId(size_t row_id) {
    base_rowid = row_id;
  }

  /*! \return Estimation of memory cost of this page. */
  static size_t EllpackPageImpl::MemCostBytes(size_t num_rows, size_t row_stride, const common::HistogramCuts&cuts) ;


  /*! \brief Return the total number of symbols (total number of bins plus 1 for
   * not found). */
  size_t NumSymbols() const { return cuts_.TotalBins() + 1; }

  EllpackDeviceAccessor GetDeviceAccessor(int device) const;

 private:
  /*!
   * \brief Compress a single page of CSR data into ELLPACK.
   *
   * @param device The GPU device to use.
   * @param row_batch The CSR page.
   */
  void CreateHistIndices(int device,
                         const SparsePage& row_batch
                         );
  /*!
   * \brief Initialize the buffer to store compressed features.
   */
  void InitCompressedData(int device);


public:
  /*! \brief Whether or not if the matrix is dense. */
  bool is_dense;
  /*! \brief Row length for ELLPack. */
  size_t row_stride;
  size_t base_rowid{0};
  size_t n_rows{};
  /*! \brief global index of histogram, which is stored in ELLPack format. */
  HostDeviceVector<common::CompressedByteT> gidx_buffer;
  common::HistogramCuts cuts_;
private:
  common::Monitor monitor_;
};

}  // namespace xgboost

#endif  // XGBOOST_DATA_ELLPACK_PAGE_H_
