/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: model dumper client implementation
 */
#ifndef HIAI_API_MODEL_DUMPER_H
#define HIAI_API_MODEL_DUMPER_H
#include "graph/graph.h"
#include "graph/graph_api_export.h"

namespace hiai {
class GRAPH_API_EXPORT ModelDumper {
public:
    static bool SaveToFile(const ge::Graph& irGraph, const std::string& file);
};
}  // namespace hiai
#endif // HIAI_API_MODEL_DUMPER_H