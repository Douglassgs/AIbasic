import os, json
import opencc

# 模块说明：
# 该脚本遍历 `chinese-poetry/` 目录下的各类诗词 JSON 文件，
# 将条目按训练/测试拆分并写入本地的 `train.txt` 与 `test.txt`，
# 同时将繁体字转换为简体字，并在每条记录末尾添加特殊结束标记 `<|endoftext|>`。

# 数据根目录
base_dir = 'chinese-poetry/'
# 需要处理的子目录列表（按仓库中文件夹命名）
classes = ['五代诗词', '元曲', '全唐诗', '四书五经', '宋词', '幽梦影', '御定全唐詩', '曹操诗集', '楚辞', '水墨唐诗',
           '纳兰性德', '蒙学', '论语', '诗经']

# 目标输出文件：训练集与测试集（写入模式）
dst_train_file = open('./train.txt', 'w')
dst_test_file = open('./test.txt', 'w')

# 繁体转简体转换器（opencc），配置为 t2s：繁体->简体
converter = opencc.OpenCC('t2s')
# 错误计数器：用于记录遇到不可处理字符或异常条目次数
error_count = 0


def write_file(item, dst_file):
    """
    将单个诗词条目写入目标文件。

    参数:
      item: dict，期望包含 'title' 与 'paragraphs'。
      dst_file: 已打开的文件对象，写入转换并拼接好的文本。

    行为:
      - 将题目与段落按行拼接为一个字符串
      - 使用 opencc 将繁体转为简体
      - 如果包含特定的异常字符（例如数据库中出现的罕见字），记录并跳过
      - 在末尾追加 `<|endoftext|>` 标记后写入文件
    """
    global error_count

    title = item['title']
    paragraphs = item['paragraphs']

    # 以题目为起始行，后续每段占一行
    content = f'\n{title}'
    for p in paragraphs:
        content = f'{content}\n{p}'

    # 繁体转简体
    content = converter.convert(content)

    # 遇到特殊无法处理字符时，计数并跳过该条（保守处理）
    if '𫗋' in content:
        print(f'{content}----')
        error_count += 1
        return

    # 添加结束标记，供后续 tokenizer/数据加载器识别样本边界
    content = content + '<|endoftext|>'
    dst_file.write(content)


def process_json(file):
    """
    处理单个 JSON 文件：解析为列表后按规则拆分为训练与测试样本并写入文件。

    规则说明：
      - 仅处理扩展名为 .json 的文件
      - 期望 JSON 文件加载后为 list，每个元素为 poem/item dict
      - 如果列表长度大于 100，则将最后一条作为测试样本，其余为训练样本；否则全部作为训练样本
    """
    if not file.endswith('.json'):
        return

    with open(file, 'r') as f:
        json_content = f.read()
        array = json.loads(json_content)
        if type(array) != list:
            # 非列表格式（例如单对象或其他结构）直接跳过
            return

        # 根据文件大小决定是否拆分测试集
        if len(array) > 100:
            train_array = array[:-1]
            test_array = array[-1:]
        else:
            train_array = array
            test_array = None

        # 写入训练样本
        for item in train_array:
            if 'title' not in item.keys() or 'paragraphs' not in item.keys():
                continue

            write_file(item, dst_train_file)

        # 写入测试样本（如果存在）
        if test_array is not None:
            for item in test_array:
                if 'title' not in item.keys() or 'paragraphs' not in item.keys():
                    continue

                write_file(item, dst_test_file)


# 遍历各分类目录，处理其中的 JSON 文件（支持两级目录结构）
for cls in classes:
    dir = base_dir + cls
    files = os.listdir(dir)

    for f in files:
        f = f'{dir}/{f}'
        if os.path.isdir(f):
            # 跳过名为 error 的子目录
            if 'error' in f:
                continue

            for ff in os.listdir(f):
                process_json(f'{f}/{ff}')
        else:
            process_json(f)

# 关闭写入句柄后重新以读取模式打开进行计数统计
dst_train_file.close()
dst_test_file.close()

dst_train_file = open('./train.txt', 'r')
dst_test_file = open('./test.txt', 'r')

train_count = 0
test_count = 0

# 统计每个文件中 `<|endoftext|>` 的出现次数作为样本数近似值
for line in dst_train_file:
    if '<|endoftext|>' in line:
        train_count += 1

for line in dst_test_file:
    if '<|endoftext|>' in line:
        test_count += 1

print(f'train_count: {train_count}, test_count: {test_count}, error_count: {error_count}')
