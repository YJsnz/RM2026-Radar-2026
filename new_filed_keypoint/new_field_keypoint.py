import open3d as o3d

# 加载3D模型
mesh = o3d.io.read_triangle_mesh("field/your_new_model.PLY")

# 可视化模型并手动选择关键点
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()
vis.add_geometry(mesh)
vis.run()  # 按空格键选择点，按Q退出
selected_points = vis.get_picked_points()

# 打印选中点的坐标
for i, idx in enumerate(selected_points):
    print(f"Point {i+1}: {mesh.vertices[idx]}")