from flask import Flask, render_template, jsonify, request
import os
import torch
import networkx as nx
import json
from utils.data_loader import load_graph_data, preprocess_data

# 创建Flask应用
app = Flask(__name__)

# 主页路由
@app.route('/')
def index():
    return render_template('index.html')

# 预处理数据路由
@app.route('/preprocess')
def preprocess():
    try:
        network_data = preprocess_data()
        return jsonify({"success": True, "message": "数据预处理成功", "stats": network_data['stats']})
    except Exception as e:
        return jsonify({"success": False, "message": f"数据预处理失败: {str(e)}"})

# 获取网络数据路由
@app.route('/get_network_data')
def get_network_data():
    try:
        with open('static/data/network_data.json', 'r') as f:
            network_data = json.load(f)
        return jsonify(network_data)
    except Exception as e:
        return jsonify({"success": False, "message": f"获取网络数据失败: {str(e)}"})

# 启动应用
if __name__ == '__main__':
    app.run(debug=True)
