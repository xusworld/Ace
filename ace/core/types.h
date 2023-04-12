#pragma once

namespace ace {

/**
 *  \brief Anakin running precision
 *   some target plantform maybe doesn't support some precision.
 */
enum class Precision : int { INT4 = -10, INT8 = -2, FP16 = -1, FP32 = 0, FP64 };

/**
 *  \brief Operator run type of operator executor.
 */
enum class OpRunType : int {
  SYNC,  ///< the net exec synchronous (for GPU, means single-stream)
  ASYNC  ///< ASYNC the net exec asynchronous (for GPU, means mutli-stream)
};

/**
 *  \brief service run pattern
 */
enum class ServiceRunPattern : int { SYNC, ASYNC };

/**
 *  \brief Inner return type used by Status type.
 */
enum class RetType {
  SUC,        ///< succeess
  ERR,        ///< error
  IMME_EXIT,  ///< need immediately exit
};

/**
 *  \brief Request type for input of operator in inference.(NOT used yet)
 *
 *   Normally operator doesn't need, except cases like NLP data or Batch Norm
 * requires.
 */
enum class EnumReqType {
  OFFSET,  ///< request hold offset info for inputs. which is used in sequence
           ///< data.
  NONE     ///< hold none data for inputs.
};

}  // namespace ace