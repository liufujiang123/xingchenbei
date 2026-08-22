// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(MhcExpand,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT16, C_DT_BF16),
    ASCENDC_TPL_UINT_DECL(MHC_MULT_KIND, ASCENDC_TPL_4_BW,
                          ASCENDC_TPL_UI_LIST, 0, 2, 4),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT16, C_DT_BF16),
        ASCENDC_TPL_UINT_SEL(MHC_MULT_KIND, ASCENDC_TPL_UI_LIST, 0, 2, 4),
    ),
);
