#ifndef CONVERTER_SOURCE_OPTIMIZER_MERGE_MERGE_HELPERS_HPP_
#define CONVERTER_SOURCE_OPTIMIZER_MERGE_MERGE_HELPERS_HPP_

#include <ace/expr/Expr.hpp>
#include <vector>

namespace ace {
namespace helpers {

bool IsConstant(Express::EXPRP expr);
bool IsBinaryOp(Express::EXPRP expr);
bool IsUnaryOp(Express::EXPRP expr);

bool IsBinaryAdd(Express::EXPRP expr);
bool IsBinarySub(Express::EXPRP expr);
bool IsBinaryMul(Express::EXPRP expr);

bool IsBinarySquaredDifference(Express::EXPRP expr);

bool IsUnarySquare(Express::EXPRP expr);
bool IsUnaryRsqrt(Express::EXPRP expr);

bool IsReductionMean(Express::EXPRP expr);

bool IsConvolution(Express::EXPRP expr);

bool IsExpandDims(Express::EXPRP expr);

Express::EXPRP InputExpr(Express::EXPRP expr, int input_index);
Express::EXPRP OutputExpr(Express::EXPRP expr, int output_index);

std::vector<Express::VARP> OutputVars(Express::EXPRP expr);

Express::VARP ConvertLayout(Express::VARP input,
                            Express::Dimensionformat dest_layout,
                            Express::Dimensionformat src_layout);

}  // namespace helpers
}  // namespace ace

#endif  // CONVERTER_SOURCE_OPTIMIZER_MERGE_MERGE_HELPERS_HPP_
