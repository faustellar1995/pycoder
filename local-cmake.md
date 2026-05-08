# 本机 CMake / 编译器 / Qt / PCL 指引

用于自用模型、Cursor Rule 或查阅：**路径随重装软件需替换**。仓库内 PCL 示例见 `pclsamples/`。

---

## 1. 工程与输出路径

| 项 | 路径 |
|----|------|
| PCL 示例源码根目录 | `d:\BaiduSyncdisk\neos\projects\pclsamples` |
| `pcl_project` 子工程 | `d:\BaiduSyncdisk\neos\projects\pclsamples\pcl_project` |
| CMake 预设 `PCL_DIR` | `D:/Lib/PCL1.15.1/cmake` |
| 运行时输出目录 | 以各目录下 `CMakeLists.txt` 为准；常见为 `D:/bin`，或分 `D:/bin/debug`、`D:/bin/release` |
| PCL All-in-One 修复脚本 | `pclsamples/scripts/fix_pcl_all_in_one_import_libs.ps1`（脚本内 `$PclRoot` 默认 `D:/Lib/PCL1.15.1`） |

---

## 2. 编译器与 CMake 生成器

### Visual Studio 2026（示例：`pclsamples/build_agent_check`）

| 项 | 路径 |
|----|------|
| 生成器 | `Visual Studio 18 2026`，`-A x64` |
| 安装实例 | `D:/Program Files/Microsoft Visual Studio/18/Community` |
| `cl.exe` | `D:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.50.35717/bin/HostX64/x64/cl.exe` |
| `link.exe` | `.../Hostx64/x64/link.exe` |
| `lib.exe` | `.../Hostx64/x64/lib.exe` |

### Visual Studio 2022（示例：`pclsamples/build_vs`）

| 项 | 路径 |
|----|------|
| 生成器 | `Visual Studio 17 2022`，`-A x64` |
| 安装实例 | `C:/Program Files/Microsoft Visual Studio/2022/Professional` |
| MSVC 工具集（示例） | `.../VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/` |

同一台机器可能有多套 VS；**同一工程**请固定一套生成器与工具集，避免与 Qt/预编译库混用冲突。

---

## 3. Windows SDK 与资源编译器

| 项 | 路径（示例） |
|----|----------------|
| `rc.exe` | `D:/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe` |

---

## 4. Qt 与 Qt Creator

| 项 | 路径 |
|----|------|
| `CMAKE_PREFIX_PATH`（Kit：Qt 5.14.2 MSVC2017 64bit） | `D:/Qt/Qt5.14.2/5.14.2/msvc2017_64` |
| `qmake` | `D:/Qt/Qt5.14.2/5.14.2/msvc2017_64/bin/qmake.exe` |
| Qt Creator 构建目录示例 | `pclsamples/build/Desktop_Qt_5_14_2_MSVC2017_64bit-Debug` / `...-Release` |
| Ninja（Qt Creator 附带集成，示例） | `D:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe` |

Qt 为 **msvc2017_64**；若使用更新版 MSVC，一般可编译，如遇运行库/ABI 问题，请在 Qt Creator 中选与 Qt 文档一致的编译器套件。

---

## 5. PCL / 第三方库与运行 PATH

| 项 | 路径 |
|----|------|
| PCL 根目录 | `D:/Lib/PCL1.15.1` |
| VTK（捆绑）CMake | `D:/Lib/PCL1.15.1/3rdParty/VTK/lib/cmake/vtk-9.4` |
| VTK 运行时 DLL | `D:/Lib/PCL1.15.1/3rdParty/VTK/bin` |
| PCL `bin` | `D:/Lib/PCL1.15.1/bin` |
| FLANN | `D:/Lib/PCL1.15.1/3rdParty/FLANN` |
| Eigen | `D:/Lib/PCL1.15.1/3rdParty/Eigen3` |
| OpenNI2（若已安装） | Lib：`C:/Program Files/OpenNI2/Lib/OpenNI2.lib`，Include：`C:/Program Files/OpenNI2/Include` |

缺 DLL 时可在 PowerShell 中：

```powershell
$env:PATH = "D:\Lib\PCL1.15.1\3rdParty\VTK\bin;D:\Lib\PCL1.15.1\bin;" + $env:PATH
```

---

## 6. 自动化：配置 → Debug 编译 → 运行

在 `pclsamples` 目录、PowerShell 中：

1. **可选**（CMake 报 VTK/FLANN 的 `.lib` 路径不存在时）：  
   `powershell -ExecutionPolicy Bypass -File "d:\BaiduSyncdisk\neos\projects\pclsamples\scripts\fix_pcl_all_in_one_import_libs.ps1"`
2. **配置**：  
   `cmake -B build_debug -S . -G "Visual Studio 18 2026" -A x64`
3. **编译 Debug**：  
   `cmake --build build_debug --config Debug`
4. **运行**：可执行文件路径以当前 `CMakeLists.txt` 为准；运行前设置上一节 `PATH`。

---

## 7. MSVC 与中文源码

- UTF-8 源码含中文时，目标上需 **`/utf-8`**（见 `pclsamples/CMakeLists.txt`），否则易出现 C2001。
- 更通用的 CMake 片段见用户 skill：`local-cmake`（`SKILL.md`）。

---

## 8. 与 skill「local-cmake」的关系

- **仓库内** `local-cmake.md`（本文件）：本机路径与 **pclsamples / PCL** 流水线备忘。
- **Cursor skill** `C:\Users\abcde\.cursor\skills\local-cmake\SKILL.md`：通用 CMake 约定、输出目录、Qt、手工库引用片段。

二者可同时保留：改路径只更新本文件即可。
