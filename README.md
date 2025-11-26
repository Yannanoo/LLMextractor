## DeepSeek-Powered Feature Extraction

This project ingests每次住院/门诊 JSON，自动向 DeepSeek Chat API 发起信息抽取，并把结果拼成既能追溯证据又能直接分析的宽表。相比初版，新脚本支持：
- 同一病人的多次就诊排序（按 `enter_time`），并把重复特征自动映射到 `Suvmax_T1`、`Suvmax_T2` ……
- 目录级批量扫描 JSON，省去一条条传参。
- 同时输出结构化 JSON（保留完整证据）和 CSV（方便建模/制表）。

### 1. 环境准备（面向 Python 新手）
1. **安装 Python 3.10**  
   - **macOS**：已安装 Homebrew 的话执行 `brew install pyenv`，然后 `pyenv install 3.10.15 && pyenv global 3.10.15`。  
   - **Windows**：从 [python.org](https://www.python.org/downloads/release/python-31015/) 下载 3.10 安装包，安装时勾选 “Add Python to PATH”。  
   - **Linux**：可用 `sudo apt install python3.10 python3.10-venv`（Ubuntu 示例）。  
   装完后在终端输入 `python3 --version`（或 `python --version`）确认输出为 `3.10.x`。

2. **获取代码**  
   把项目文件夹（例如 `Rmaintain`）拷到任意目录，或在 git 环境下 `git clone <repo>`。

3. **创建隔离环境**  
   ```bash
   cd /Users/yannan/PycharmProjects/Rmaintain  # 换成你的路径
   python3 -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   ```
   激活后终端前缀会出现 `(.venv)`。

4. **安装依赖**  
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **配置 DeepSeek API Key**  
   - macOS / Linux：`export DEEPSEEK_API_KEY="sk-..."`  
   - Windows PowerShell：`$env:DEEPSEEK_API_KEY="sk-..."`  
   (可写入 `~/.zshrc`、PowerShell Profile 等以便永久生效。)

### 2. 关键文件
- `extract_features.py`：主入口；负责遍历 JSON、构造 prompt、调用 DeepSeek、聚合结果并落地 CSV/JSON。
- `features_config.json`：特征配置。每条包含 `name`（提示给 LLM）和 `alias`（CSV 列名前缀）。可随时增删或改别名。
- `1151005698843348992.json`：示例住院记录，便于本地验证。

### 3. 运行示例
```shell
cd /Users/yannan/PycharmProjects/Rmaintain
python extract_features.py \
  --patient-dir /Users/yannan/PycharmProjects/Rmaintain \
  --config /Users/yannan/PycharmProjects/Rmaintain/features_config.json \
  --output-json /Users/yannan/PycharmProjects/Rmaintain/all_visits.json \
  --output-csv /Users/yannan/PycharmProjects/Rmaintain/all_visits.csv
```
常用参数：
- `--patient-file /abs/path/a.json` （可重复）或 `--patient-dir /abs/path/dir`（递归扫描）。
- `--omit-evidence` 如果 CSV 不需要 `*_evidence` 列。
- `--retries / --retry-backoff / --temperature / --model` 对 DeepSeek 请求做自定义。

运行后会生成两个产物：
1. `all_visits.json`：以 `patient_id` 为键，列出所有 visit（包含 visit 次序、原 JSON 路径、每个特征的 `value`+`evidence`）。
2. `all_visits.csv`：宽表，默认列顺序为
   ```
   patient_id, patient_name, total_visits,
   visit_id_T1, visit_time_T1, Suvmax_T1, Suvmax_T1_evidence, ...,
   visit_id_T2, visit_time_T2, Suvmax_T2, ...
   ```
   若某病人只有两次就诊，就不会出现 `*_T3` 列的值（保持为空）。

### 4. 关于特征配置
`features_config.json` 格式示例：
```json
{
  "continuous_features": [
    { "name": "SUVmax (首次治疗)", "alias": "Suvmax" }
  ],
  "categorical_features": [
    { "name": "治疗时B症状(0:无; 1:有)", "alias": "BSymptom" }
  ]
}
```
- `name` 是 prompt 中的原文，尽量贴近临床描述。
- `alias` 会被用作 CSV 列名前缀（并自动加上 `_T{n}`）。若省略 `alias`，脚本会依据 `name` 生成 CamelCase，但为了保证和业务口径一致，建议显式填写。
- 想新增特征时，只需在对应数组里新增对象即可，脚本会自动调整 prompt 以及 CSV 列。

### 5. 多次就诊的命名规则
- 就诊按 `enter_time` 排序（无法解析时间时使用文件名的字典序兜底）。
- 每个特征都会生成若干列：`<alias>_T1`, `<alias>_T1_evidence`, `<alias>_T2`, ...
- “证据”列保留 DeepSeek 引用的原句，方便人工复核；若不需要可加 `--omit-evidence`。

### 6. 错误处理与重试
- 请求 DeepSeek 失败或解析异常会自动重试 3 次（可通过 `--retries` 和 `--retry-backoff` 调整）。
- 每次出错都会在异常信息里写入对应的 JSON 路径，便于定位。
- 如果需要代理/自建网关，可用 `--api-url` 改写基础 URL。

### 7. 二次开发建议
- 在 `merge_text` 中插入自定义排序逻辑或过滤，减少 prompt token。
- `aggregate_patients` 中可以把 prompt/response 原文一并落盘，满足审计留痕。
- 若批量次数多，可在外层加缓存，把相同 visit 的结果复用，节约 token。

> 运行前请确保 `DEEPSEEK_API_KEY` 有效、网络能访问 DeepSeek API。
