import random

# 假设你有一组标签
labels = ["ClassA", "ClassB", "ClassC", "ClassD"]

# 为每个标签生成随机颜色
color_map = {}
for label in labels:
    # 生成一个随机的RGB颜色
    color = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color_map[label] = color

# 输出颜色映射
print(color_map)
