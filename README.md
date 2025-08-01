# 3DGS(VULKAN)

## 工程目录
```
├── entry/src/main	         // 代码区
│  ├── cpp
│  │  ├── types
│  │  │  ├── libnativerender
             └── index.d.ts      // native层接口注册文件
│  │  │── napi_init.cpp          // native api层接口的具体实现函数
│  │  │── CMakeLists.txt         // native层编译配置
│  │  │── 3rdParty               // 三方件
│  │  │── commom                 // 通用接口
│  │  │── file                   // 文件管理
│  │  │── libs                   // 三方动态库
│  │  │── manager                // native&arkts交互
│  │  │── render                 // 渲染
│  │  │── vulkanbase             // vulkan基础能力封装
│  ├── ets
│  │  ├── entryability
             └── EntryAbility.ts // 程序入口类
│  │  ├── pages
             └── index.ets       // 主界面展示类
│  ├── resources                 // 资源文件目录
│  │  ├── base
│  │  │  ├── media
             └── icon.png        // 图片资源
│  │  ├── rawfile
│  │  │  ├── Sponza
             └── sponza.obj     // 模型资源
```


## 注意事项！！！！
1. human/smpL/src/smpl/smpl_male.cpp 这个文件过大无法上传, 找朱嘉禾或者曾畅
2. 是否需要单独出物体，通过model_compute.h中类成员变量isObject设置



